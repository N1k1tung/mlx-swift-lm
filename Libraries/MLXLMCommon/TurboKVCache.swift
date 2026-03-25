// Copyright © 2026 Apple Inc.

import Foundation
import MLX

public let defaultTurboQuantSeed = 0
private let _turboQuantEps: Float = 1e-6
private let _polarMaxLevels = 4

enum _TurboQuantCodecMode: String {
    case mse
    case prod
}

struct TurboQuantMSEState {
    var norms: MLXArray
    var indices: MLXArray
}

struct TurboQuantProdState {
    var norms: MLXArray
    var mseIndices: MLXArray
    var residualNorms: MLXArray
    var qjlSigns: MLXArray
}

struct TurboQuantPolarState {
    var radii: MLXArray
    var levelIndices: [MLXArray]
}

struct TurboQuantPolarProdState {
    var norms: MLXArray
    var polarState: TurboQuantPolarState
    var residualNorms: MLXArray
    var qjlSigns: MLXArray
}

indirect enum TurboQuantState {
    case mse(TurboQuantMSEState)
    case prod(TurboQuantProdState)
    case polar(TurboQuantPolarState)
    case polarProd(TurboQuantPolarProdState)
    case split(TurboQuantSplitState)
}

struct TurboQuantSplitState {
    var low: TurboQuantState
    var high: TurboQuantState
}

indirect enum TurboQuantPreparedQueries {
    case array(MLXArray)
    case pair(MLXArray, MLXArray)
    case split(TurboQuantPreparedQueries, TurboQuantPreparedQueries)
}

private enum _TurboQuantRegistry {
    static let lock = NSLock()
    static var rotations: [String: MLXArray] = [:]
    static var projections: [String: MLXArray] = [:]
    static var codebooks: [String: MLXArray] = [:]
    static var polarCodebooks: [String: MLXArray] = [:]
    static var kernels: [String: MLXFast.MLXFastKernel] = [:]
    static var compiledIntegerDecoders: [Int: @Sendable ([MLXArray]) -> [MLXArray]] = [:]
}

private func _metalAvailable() -> Bool {
    Device.defaultDevice().deviceType == .gpu
}

private func _cachedKernel(
    key: String,
    build: () -> MLXFast.MLXFastKernel
) -> MLXFast.MLXFastKernel {
    _TurboQuantRegistry.lock.lock()
    defer { _TurboQuantRegistry.lock.unlock() }
    if let kernel = _TurboQuantRegistry.kernels[key] {
        return kernel
    }
    let kernel = build()
    _TurboQuantRegistry.kernels[key] = kernel
    return kernel
}

private func _mseScoreKernel() -> MLXFast.MLXFastKernel? {
    guard _metalAvailable() else { return nil }
    let source = #"""
        auto lane = thread_position_in_grid.x;
        auto repeat_idx = thread_position_in_grid.y;
        auto n = thread_position_in_grid.z;

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto repeat_count = q_rot_shape[2];
        if (repeat_idx >= repeat_count) {
            return;
        }

        auto b = n / (kv_heads * token_count);
        auto rem = n % (kv_heads * token_count);
        auto h = rem / token_count;
        auto t = rem % token_count;

        auto q_ptr = q_rot + ((b * kv_heads + h) * repeat_count + repeat_idx) * Dim;
        auto packed_ptr = packed + ((b * kv_heads + h) * token_count + t) * PackedWidth;

        float acc = 0.0f;
        for (int d = lane; d < Dim; d += 32) {
            int bit_offset = d * Bits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = packed_ptr[word_idx] >> offset;
            int spill = offset + Bits - 32;
            if (spill > 0) {
                value |= packed_ptr[word_idx + 1] << (Bits - spill);
            }
            value &= ((1u << Bits) - 1u);
            acc += static_cast<float>(q_ptr[d]) * codebook[value];
        }

        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            out[((b * kv_heads + h) * repeat_count + repeat_idx) * token_count + t] =
                acc * static_cast<float>(norms[(b * kv_heads + h) * token_count + t]);
        }
    """#
    return _cachedKernel(key: "turboquant_mse_score") {
        MLXFast.metalKernel(
            name: "turboquant_mse_score",
            inputNames: ["q_rot", "norms", "packed", "codebook"],
            outputNames: ["out"],
            source: source
        )
    }
}

private func _packLowbitKernel() -> MLXFast.MLXFastKernel? {
    guard _metalAvailable() else { return nil }
    let source = #"""
        auto word = thread_position_in_grid.x;
        auto row = thread_position_in_grid.y;

        if (row >= values_shape[0] || word >= PackedWidth) {
            return;
        }

        auto values_ptr = values + row * Length;
        uint packed_word = 0u;
        int start = max(0, (int(word) * 32 - (Bits - 1)) / Bits);
        int end = min(Length, ((int(word) + 1) * 32 + (Bits - 1)) / Bits);

        for (int idx = start; idx < end; ++idx) {
            int bit_offset = idx * Bits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = values_ptr[idx] & ((1u << Bits) - 1u);
            if (word_idx == word) {
                packed_word |= value << offset;
            }
            if (word_idx + 1 == word) {
                int spill = offset + Bits - 32;
                if (spill > 0) {
                    packed_word |= value >> (Bits - spill);
                }
            }
        }

        out[row * PackedWidth + word] = packed_word;
    """#
    return _cachedKernel(key: "turboquant_pack_lowbit") {
        MLXFast.metalKernel(
            name: "turboquant_pack_lowbit",
            inputNames: ["values"],
            outputNames: ["out"],
            source: source
        )
    }
}

private func _unpackLowbitKernel() -> MLXFast.MLXFastKernel? {
    guard _metalAvailable() else { return nil }
    let source = #"""
        auto idx = thread_position_in_grid.x;
        auto row = thread_position_in_grid.y;

        if (row >= packed_shape[0] || idx >= Length) {
            return;
        }

        auto packed_ptr = packed + row * PackedWidth;
        int bit_offset = idx * Bits;
        int word_idx = bit_offset / 32;
        int offset = bit_offset % 32;
        uint value = packed_ptr[word_idx] >> offset;
        int spill = offset + Bits - 32;
        if (spill > 0) {
            value |= packed_ptr[word_idx + 1] << (Bits - spill);
        }
        out[row * Length + idx] = value & ((1u << Bits) - 1u);
    """#
    return _cachedKernel(key: "turboquant_unpack_lowbit") {
        MLXFast.metalKernel(
            name: "turboquant_unpack_lowbit",
            inputNames: ["packed"],
            outputNames: ["out"],
            source: source
        )
    }
}

private func _qjlScoreKernel() -> MLXFast.MLXFastKernel? {
    guard _metalAvailable() else { return nil }
    let source = #"""
        auto lane = thread_position_in_grid.x;
        auto repeat_idx = thread_position_in_grid.y;
        auto n = thread_position_in_grid.z;

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto repeat_count = q_proj_shape[2];
        if (repeat_idx >= repeat_count) {
            return;
        }

        auto b = n / (kv_heads * token_count);
        auto rem = n % (kv_heads * token_count);
        auto h = rem / token_count;
        auto t = rem % token_count;

        auto q_ptr = q_proj + ((b * kv_heads + h) * repeat_count + repeat_idx) * Dim;
        auto packed_ptr = signs + ((b * kv_heads + h) * token_count + t) * PackedWidth;

        float acc = 0.0f;
        for (int d = lane; d < Dim; d += 32) {
            int word_idx = d / 32;
            int offset = d % 32;
            uint bit = (packed_ptr[word_idx] >> offset) & 1u;
            float sign = bit ? 1.0f : -1.0f;
            acc += static_cast<float>(q_ptr[d]) * sign;
        }

        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            auto idx = (b * kv_heads + h) * token_count + t;
            out[((b * kv_heads + h) * repeat_count + repeat_idx) * token_count + t] =
                acc
                * static_cast<float>(norms[idx])
                * static_cast<float>(residual_norms[idx])
                * scale[0];
        }
    """#
    return _cachedKernel(key: "turboquant_qjl_score") {
        MLXFast.metalKernel(
            name: "turboquant_qjl_score",
            inputNames: ["q_proj", "norms", "residual_norms", "signs", "scale"],
            outputNames: ["out"],
            source: source
        )
    }
}

private func _prodScoreKernel() -> MLXFast.MLXFastKernel? {
    guard _metalAvailable() else { return nil }
    let source = #"""
        auto lane = thread_position_in_grid.x;
        auto repeat_idx = thread_position_in_grid.y;
        auto n = thread_position_in_grid.z;

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto repeat_count = q_rot_shape[2];
        if (repeat_idx >= repeat_count) {
            return;
        }

        auto b = n / (kv_heads * token_count);
        auto rem = n % (kv_heads * token_count);
        auto h = rem / token_count;
        auto t = rem % token_count;

        auto q_rot_ptr = q_rot + ((b * kv_heads + h) * repeat_count + repeat_idx) * Dim;
        auto q_proj_ptr = q_proj + ((b * kv_heads + h) * repeat_count + repeat_idx) * Dim;
        auto mse_ptr = mse_packed + ((b * kv_heads + h) * token_count + t) * MsePackedWidth;
        auto sign_ptr = signs + ((b * kv_heads + h) * token_count + t) * SignPackedWidth;

        float mse_acc = 0.0f;
        float qjl_acc = 0.0f;
        for (int d = lane; d < Dim; d += 32) {
            int bit_offset = d * MseBits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = mse_ptr[word_idx] >> offset;
            int spill = offset + MseBits - 32;
            if (spill > 0) {
                value |= mse_ptr[word_idx + 1] << (MseBits - spill);
            }
            value &= ((1u << MseBits) - 1u);
            mse_acc += static_cast<float>(q_rot_ptr[d]) * codebook[value];

            int sign_word = d / 32;
            int sign_offset = d % 32;
            uint bit = (sign_ptr[sign_word] >> sign_offset) & 1u;
            float sign = bit ? 1.0f : -1.0f;
            qjl_acc += static_cast<float>(q_proj_ptr[d]) * sign;
        }

        mse_acc = simd_sum(mse_acc);
        qjl_acc = simd_sum(qjl_acc);
        if (thread_index_in_simdgroup == 0) {
            auto idx = (b * kv_heads + h) * token_count + t;
            out[((b * kv_heads + h) * repeat_count + repeat_idx) * token_count + t] =
                static_cast<float>(norms[idx]) * (
                    mse_acc
                    + scale[0] * static_cast<float>(residual_norms[idx]) * qjl_acc
                );
        }
    """#
    return _cachedKernel(key: "turboquant_prod_score") {
        MLXFast.metalKernel(
            name: "turboquant_prod_score",
            inputNames: [
                "q_rot",
                "q_proj",
                "norms",
                "residual_norms",
                "mse_packed",
                "signs",
                "codebook",
                "scale",
            ],
            outputNames: ["out"],
            source: source
        )
    }
}

private func _prodScoreMultiKernel() -> MLXFast.MLXFastKernel? {
    guard _metalAvailable() else { return nil }
    let source = #"""
        auto lane = thread_position_in_grid.x;
        auto n = thread_position_in_grid.z;

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];

        auto b = n / (kv_heads * token_count);
        auto rem = n % (kv_heads * token_count);
        auto h = rem / token_count;
        auto t = rem % token_count;

        auto q_rot_base = q_rot + ((b * kv_heads + h) * RepeatCount) * Dim;
        auto q_proj_base = q_proj + ((b * kv_heads + h) * RepeatCount) * Dim;
        auto mse_ptr = mse_packed + ((b * kv_heads + h) * token_count + t) * MsePackedWidth;
        auto sign_ptr = signs + ((b * kv_heads + h) * token_count + t) * SignPackedWidth;

        float mse_acc[RepeatCount];
        float qjl_acc[RepeatCount];
        for (int r = 0; r < RepeatCount; ++r) {
            mse_acc[r] = 0.0f;
            qjl_acc[r] = 0.0f;
        }

        for (int d = lane; d < Dim; d += 32) {
            int bit_offset = d * MseBits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = mse_ptr[word_idx] >> offset;
            int spill = offset + MseBits - 32;
            if (spill > 0) {
                value |= mse_ptr[word_idx + 1] << (MseBits - spill);
            }
            value &= ((1u << MseBits) - 1u);
            float code = codebook[value];

            int sign_word = d / 32;
            int sign_offset = d % 32;
            uint bit = (sign_ptr[sign_word] >> sign_offset) & 1u;
            float sign = bit ? 1.0f : -1.0f;

            for (int r = 0; r < RepeatCount; ++r) {
                mse_acc[r] += static_cast<float>(q_rot_base[r * Dim + d]) * code;
                qjl_acc[r] += static_cast<float>(q_proj_base[r * Dim + d]) * sign;
            }
        }

        for (int r = 0; r < RepeatCount; ++r) {
            mse_acc[r] = simd_sum(mse_acc[r]);
            qjl_acc[r] = simd_sum(qjl_acc[r]);
        }

        if (thread_index_in_simdgroup == 0) {
            auto idx = (b * kv_heads + h) * token_count + t;
            float norm = static_cast<float>(norms[idx]);
            float residual_norm = static_cast<float>(residual_norms[idx]);
            for (int r = 0; r < RepeatCount; ++r) {
                out[((b * kv_heads + h) * RepeatCount + r) * token_count + t] =
                    norm * (mse_acc[r] + scale[0] * residual_norm * qjl_acc[r]);
            }
        }
    """#
    return _cachedKernel(key: "turboquant_prod_score_multi") {
        MLXFast.metalKernel(
            name: "turboquant_prod_score_multi",
            inputNames: [
                "q_rot",
                "q_proj",
                "norms",
                "residual_norms",
                "mse_packed",
                "signs",
                "codebook",
                "scale",
            ],
            outputNames: ["out"],
            source: source
        )
    }
}

private func _mseWeightedRotKernel() -> MLXFast.MLXFastKernel? {
    guard _metalAvailable() else { return nil }
    let source = #"""
        auto lane = thread_position_in_grid.x;
        auto dim_idx = thread_position_in_grid.y;
        auto n = thread_position_in_grid.z;

        if (dim_idx >= Dim) {
            return;
        }

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto repeat_count = weights_shape[2];
        auto b = n / (kv_heads * repeat_count);
        auto rem = n % (kv_heads * repeat_count);
        auto h = rem / repeat_count;
        auto repeat_idx = rem % repeat_count;

        auto weights_ptr = weights + ((b * kv_heads + h) * repeat_count + repeat_idx) * token_count;
        auto norms_ptr = norms + (b * kv_heads + h) * token_count;
        auto packed_ptr = packed + ((b * kv_heads + h) * token_count) * PackedWidth;

        float acc = 0.0f;
        for (int t = lane; t < token_count; t += 32) {
            auto token_ptr = packed_ptr + t * PackedWidth;
            int bit_offset = dim_idx * Bits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = token_ptr[word_idx] >> offset;
            int spill = offset + Bits - 32;
            if (spill > 0) {
                value |= token_ptr[word_idx + 1] << (Bits - spill);
            }
            value &= ((1u << Bits) - 1u);
            acc += static_cast<float>(weights_ptr[t])
                * static_cast<float>(norms_ptr[t])
                * codebook[value];
        }

        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            out[((b * kv_heads + h) * repeat_count + repeat_idx) * Dim + dim_idx] = acc;
        }
    """#
    return _cachedKernel(key: "turboquant_mse_weighted_rot") {
        MLXFast.metalKernel(
            name: "turboquant_mse_weighted_rot",
            inputNames: ["weights", "norms", "packed", "codebook"],
            outputNames: ["out"],
            source: source
        )
    }
}

private func _mseWeightedRotMultiKernel() -> MLXFast.MLXFastKernel? {
    guard _metalAvailable() else { return nil }
    let source = #"""
        auto lane = thread_position_in_grid.x;
        auto dim_idx = thread_position_in_grid.y;
        auto n = thread_position_in_grid.z;

        if (dim_idx >= Dim) {
            return;
        }

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto b = n / kv_heads;
        auto h = n % kv_heads;

        auto weights_base =
            weights + ((b * kv_heads + h) * RepeatCount) * token_count;
        auto norms_ptr = norms + (b * kv_heads + h) * token_count;
        auto packed_ptr = packed + ((b * kv_heads + h) * token_count) * PackedWidth;

        float acc[RepeatCount];
        for (int r = 0; r < RepeatCount; ++r) {
            acc[r] = 0.0f;
        }

        int bit_offset = dim_idx * Bits;
        int word_idx = bit_offset / 32;
        int offset = bit_offset % 32;

        for (int t = lane; t < token_count; t += 32) {
            auto token_ptr = packed_ptr + t * PackedWidth;
            uint value = token_ptr[word_idx] >> offset;
            int spill = offset + Bits - 32;
            if (spill > 0) {
                value |= token_ptr[word_idx + 1] << (Bits - spill);
            }
            value &= ((1u << Bits) - 1u);
            float code = codebook[value];
            float norm = static_cast<float>(norms_ptr[t]);
            for (int r = 0; r < RepeatCount; ++r) {
                acc[r] += static_cast<float>(weights_base[r * token_count + t]) * norm * code;
            }
        }

        for (int r = 0; r < RepeatCount; ++r) {
            acc[r] = simd_sum(acc[r]);
        }

        if (thread_index_in_simdgroup == 0) {
            for (int r = 0; r < RepeatCount; ++r) {
                out[((b * kv_heads + h) * RepeatCount + r) * Dim + dim_idx] =
                    acc[r];
            }
        }
    """#
    return _cachedKernel(key: "turboquant_mse_weighted_rot_multi") {
        MLXFast.metalKernel(
            name: "turboquant_mse_weighted_rot_multi",
            inputNames: ["weights", "norms", "packed", "codebook"],
            outputNames: ["out"],
            source: source
        )
    }
}

private func _prodScoreRepeatKernel(_ repeatCount: Int) -> MLXFast.MLXFastKernel? {
    guard _metalAvailable(), repeatCount > 1 else { return nil }
    let key = "turboquant_prod_score_repeat_\(repeatCount)"
    return _cachedKernel(key: key) {
        var lines = [
            "        auto lane = thread_position_in_grid.x;",
            "        auto n = thread_position_in_grid.z;",
            "",
            "        auto token_count = norms_shape[2];",
            "        auto kv_heads = norms_shape[1];",
            "        auto repeat_count = q_rot_shape[2];",
            "",
            "        auto b = n / (kv_heads * token_count);",
            "        auto rem = n % (kv_heads * token_count);",
            "        auto h = rem / token_count;",
            "        auto t = rem % token_count;",
            "",
            "        auto q_rot_base = q_rot + ((b * kv_heads + h) * repeat_count) * Dim;",
            "        auto q_proj_base = q_proj + ((b * kv_heads + h) * repeat_count) * Dim;",
            "        auto mse_ptr = mse_packed + ((b * kv_heads + h) * token_count + t) * MsePackedWidth;",
            "        auto sign_ptr = signs + ((b * kv_heads + h) * token_count + t) * SignPackedWidth;",
            "",
            "        auto idx = (b * kv_heads + h) * token_count + t;",
            "        float norm = static_cast<float>(norms[idx]);",
            "        float residual_norm = static_cast<float>(residual_norms[idx]);",
            "",
        ]
        for r in 0 ..< repeatCount {
            lines.append("        float mse_acc_\(r) = 0.0f;")
            lines.append("        float qjl_acc_\(r) = 0.0f;")
        }
        lines += [
            "",
            "        for (int d = lane; d < Dim; d += 32) {",
            "            int bit_offset = d * MseBits;",
            "            int word_idx = bit_offset / 32;",
            "            int offset = bit_offset % 32;",
            "            uint value = mse_ptr[word_idx] >> offset;",
            "            int spill = offset + MseBits - 32;",
            "            if (spill > 0) {",
            "                value |= mse_ptr[word_idx + 1] << (MseBits - spill);",
            "            }",
            "            value &= ((1u << MseBits) - 1u);",
            "            float code = codebook[value];",
            "",
            "            int sign_word = d / 32;",
            "            int sign_offset = d % 32;",
            "            uint bit = (sign_ptr[sign_word] >> sign_offset) & 1u;",
            "            float sign = bit ? 1.0f : -1.0f;",
            "",
        ]
        for r in 0 ..< repeatCount {
            lines.append(
                "            mse_acc_\(r) += static_cast<float>(q_rot_base[\(r) * Dim + d]) * code;"
            )
            lines.append(
                "            qjl_acc_\(r) += static_cast<float>(q_proj_base[\(r) * Dim + d]) * sign;"
            )
        }
        lines += [
            "        }",
            "",
        ]
        for r in 0 ..< repeatCount {
            lines.append("        float mse_sum_\(r) = simd_sum(mse_acc_\(r));")
            lines.append("        float qjl_sum_\(r) = simd_sum(qjl_acc_\(r));")
        }
        lines += [
            "",
            "        if (thread_index_in_simdgroup == 0) {",
        ]
        for r in 0 ..< repeatCount {
            lines.append(
                "            out[((b * kv_heads + h) * repeat_count + \(r)) * token_count + t] ="
            )
            lines.append(
                "                norm * (mse_sum_\(r) + scale[0] * residual_norm * qjl_sum_\(r));"
            )
        }
        lines += [
            "        }",
        ]
        return MLXFast.metalKernel(
            name: "turboquant_prod_score_repeat_\(repeatCount)",
            inputNames: [
                "q_rot",
                "q_proj",
                "norms",
                "residual_norms",
                "mse_packed",
                "signs",
                "codebook",
                "scale",
            ],
            outputNames: ["out"],
            source: lines.joined(separator: "\n")
        )
    }
}

private func _polarProdScoreKernel(_ levelBits: [Int]) -> MLXFast.MLXFastKernel? {
    guard _metalAvailable(), !levelBits.isEmpty else { return nil }
    let key = "turboquant_polar_prod_score_" + levelBits.map(String.init).joined(separator: "_")
    return _cachedKernel(key: key) {
        var inputNames = ["q_rot", "norms", "radii"]
        for level in 0 ..< levelBits.count {
            inputNames.append("angles_\(level + 1)")
        }
        for level in 0 ..< levelBits.count {
            inputNames.append("cos_\(level + 1)")
            inputNames.append("sin_\(level + 1)")
        }

        var lines = [
            "        auto lane = thread_position_in_grid.x;",
            "        auto repeat_idx = thread_position_in_grid.y;",
            "        auto n = thread_position_in_grid.z;",
            "",
            "        auto token_count = norms_shape[2];",
            "        auto kv_heads = norms_shape[1];",
            "        auto repeat_count = q_rot_shape[2];",
            "        if (repeat_idx >= repeat_count) {",
            "            return;",
            "        }",
            "",
            "        auto b = n / (kv_heads * token_count);",
            "        auto rem = n % (kv_heads * token_count);",
            "        auto h = rem / token_count;",
            "        auto t = rem % token_count;",
            "",
            "        auto q_ptr = q_rot + ((b * kv_heads + h) * repeat_count + repeat_idx) * Dim;",
            "        auto radii_ptr = radii + ((b * kv_heads + h) * token_count + t) * BlockCount;",
            "",
            "        float acc = 0.0f;",
            "        for (int d = lane; d < Dim; d += 32) {",
            "            int block_idx = d >> Levels;",
            "            float coeff = static_cast<float>(radii_ptr[block_idx]);",
            "",
        ]

        for (index, bits) in levelBits.enumerated() {
            let level = index + 1
            let mask = (1 << bits) - 1
            lines += [
                "            auto angle_ptr_\(level) = angles_\(level) + ((b * kv_heads + h) * token_count + t) * PackedWidth\(level);",
                "            int angle_idx_\(level) = d >> \(level);",
                "            int bit_offset_\(level) = angle_idx_\(level) * \(bits);",
                "            int word_idx_\(level) = bit_offset_\(level) / 32;",
                "            int offset_\(level) = bit_offset_\(level) % 32;",
                "            uint value_\(level) = angle_ptr_\(level)[word_idx_\(level)] >> offset_\(level);",
                "            int spill_\(level) = offset_\(level) + \(bits) - 32;",
                "            if (spill_\(level) > 0) {",
                "                value_\(level) |= angle_ptr_\(level)[word_idx_\(level) + 1] << (\(bits) - spill_\(level));",
                "            }",
                "            value_\(level) &= \(mask)u;",
                "            bool use_sin_\(level) = ((d >> \(level - 1)) & 1) != 0;",
                "            coeff *= use_sin_\(level) ? static_cast<float>(sin_\(level)[value_\(level)]) : static_cast<float>(cos_\(level)[value_\(level)]);",
                "",
            ]
        }

        lines += [
            "            acc += static_cast<float>(q_ptr[d]) * coeff;",
            "        }",
            "",
            "        acc = simd_sum(acc);",
            "        if (thread_index_in_simdgroup == 0) {",
            "            auto idx = (b * kv_heads + h) * token_count + t;",
            "            out[((b * kv_heads + h) * repeat_count + repeat_idx) * token_count + t] =",
            "                acc * static_cast<float>(norms[idx]);",
            "        }",
        ]

        return MLXFast.metalKernel(
            name: key,
            inputNames: inputNames,
            outputNames: ["out"],
            source: lines.joined(separator: "\n")
        )
    }
}

private func _polarTurboScoreRepeatKernel(
    levelBits: [Int], repeatCount: Int
) -> MLXFast.MLXFastKernel? {
    guard _metalAvailable(), levelBits.count == 4, repeatCount > 0 else { return nil }
    let key =
        "turboquant_polar_turbo_score_"
        + levelBits.map(String.init).joined(separator: "_")
        + "_repeat_\(repeatCount)"
    return _cachedKernel(key: key) {
        let bits1 = levelBits[0]
        let bits2 = levelBits[1]
        let bits3 = levelBits[2]
        let bits4 = levelBits[3]
        let mask1 = (1 << bits1) - 1
        let mask2 = (1 << bits2) - 1
        let mask3 = (1 << bits3) - 1
        let mask4 = (1 << bits4) - 1

        var inputNames = ["q_rot", "q_proj", "norms", "radii"]
        for level in 0 ..< 4 { inputNames.append("angles_\(level + 1)") }
        inputNames += ["residual_norms", "signs", "scale"]
        for level in 0 ..< 4 {
            inputNames.append("cos_\(level + 1)")
            inputNames.append("sin_\(level + 1)")
        }

        var lines = [
            "        auto lane = thread_position_in_grid.x;",
            "        auto n = thread_position_in_grid.z;",
            "",
            "        auto token_count = norms_shape[2];",
            "        auto kv_heads = norms_shape[1];",
            "",
            "        auto b = n / (kv_heads * token_count);",
            "        auto rem = n % (kv_heads * token_count);",
            "        auto h = rem / token_count;",
            "        auto t = rem % token_count;",
            "",
            "        auto q_rot_base = q_rot + ((b * kv_heads + h) * RepeatCount) * Dim;",
            "        auto q_proj_base = q_proj + ((b * kv_heads + h) * RepeatCount) * Dim;",
            "        auto radii_ptr = radii + ((b * kv_heads + h) * token_count + t) * BlockCount;",
            "        auto angle1_ptr = angles_1 + ((b * kv_heads + h) * token_count + t) * PackedWidth1;",
            "        auto angle2_ptr = angles_2 + ((b * kv_heads + h) * token_count + t) * PackedWidth2;",
            "        auto angle3_ptr = angles_3 + ((b * kv_heads + h) * token_count + t) * PackedWidth3;",
            "        auto angle4_ptr = angles_4 + ((b * kv_heads + h) * token_count + t) * PackedWidth4;",
            "        auto sign_ptr = signs + ((b * kv_heads + h) * token_count + t) * SignPackedWidth;",
            "",
            "        auto idx = (b * kv_heads + h) * token_count + t;",
            "        float norm = static_cast<float>(norms[idx]);",
            "        float residual_norm = static_cast<float>(residual_norms[idx]);",
            "",
        ]
        for r in 0 ..< repeatCount {
            lines.append("        threadgroup float level1_\(r)[Dim / 2];")
            lines.append("        threadgroup float level2_\(r)[Dim / 4];")
            lines.append("        threadgroup float level3_\(r)[Dim / 8];")
            lines.append("        threadgroup float level4_\(r)[BlockCount];")
        }
        lines.append("")
        for r in 0 ..< repeatCount {
            lines.append("        float qjl_acc_\(r) = 0.0f;")
        }
        lines += [
            "",
            "        for (int pair_idx = lane; pair_idx < Dim / 2; pair_idx += 32) {",
            "            int bit_offset_1 = pair_idx * \(bits1);",
            "            int word_idx_1 = bit_offset_1 / 32;",
            "            int offset_1 = bit_offset_1 % 32;",
            "            uint value_1 = angle1_ptr[word_idx_1] >> offset_1;",
            "            int spill_1 = offset_1 + \(bits1) - 32;",
            "            if (spill_1 > 0) {",
            "                value_1 |= angle1_ptr[word_idx_1 + 1] << (\(bits1) - spill_1);",
            "            }",
            "            value_1 &= \(mask1)u;",
            "            float cos_1_val = static_cast<float>(cos_1[value_1]);",
            "            float sin_1_val = static_cast<float>(sin_1[value_1]);",
            "            int d0 = pair_idx << 1;",
            "            int d1 = d0 + 1;",
            "",
        ]
        for r in 0 ..< repeatCount {
            lines.append(
                "            level1_\(r)[pair_idx] = static_cast<float>(q_rot_base[\(r) * Dim + d0]) * cos_1_val + static_cast<float>(q_rot_base[\(r) * Dim + d1]) * sin_1_val;"
            )
        }
        lines += [
            "        }",
            "        threadgroup_barrier(mem_flags::mem_threadgroup);",
            "",
            "        for (int pair_idx = lane; pair_idx < Dim / 4; pair_idx += 32) {",
            "            int bit_offset_2 = pair_idx * \(bits2);",
            "            int word_idx_2 = bit_offset_2 / 32;",
            "            int offset_2 = bit_offset_2 % 32;",
            "            uint value_2 = angle2_ptr[word_idx_2] >> offset_2;",
            "            int spill_2 = offset_2 + \(bits2) - 32;",
            "            if (spill_2 > 0) {",
            "                value_2 |= angle2_ptr[word_idx_2 + 1] << (\(bits2) - spill_2);",
            "            }",
            "            value_2 &= \(mask2)u;",
            "            float cos_2_val = static_cast<float>(cos_2[value_2]);",
            "            float sin_2_val = static_cast<float>(sin_2[value_2]);",
            "            int child = pair_idx << 1;",
            "",
        ]
        for r in 0 ..< repeatCount {
            lines.append(
                "            level2_\(r)[pair_idx] = level1_\(r)[child] * cos_2_val + level1_\(r)[child + 1] * sin_2_val;"
            )
        }
        lines += [
            "        }",
            "        threadgroup_barrier(mem_flags::mem_threadgroup);",
            "",
            "        for (int pair_idx = lane; pair_idx < Dim / 8; pair_idx += 32) {",
            "            int bit_offset_3 = pair_idx * \(bits3);",
            "            int word_idx_3 = bit_offset_3 / 32;",
            "            int offset_3 = bit_offset_3 % 32;",
            "            uint value_3 = angle3_ptr[word_idx_3] >> offset_3;",
            "            int spill_3 = offset_3 + \(bits3) - 32;",
            "            if (spill_3 > 0) {",
            "                value_3 |= angle3_ptr[word_idx_3 + 1] << (\(bits3) - spill_3);",
            "            }",
            "            value_3 &= \(mask3)u;",
            "            float cos_3_val = static_cast<float>(cos_3[value_3]);",
            "            float sin_3_val = static_cast<float>(sin_3[value_3]);",
            "            int child = pair_idx << 1;",
            "",
        ]
        for r in 0 ..< repeatCount {
            lines.append(
                "            level3_\(r)[pair_idx] = level2_\(r)[child] * cos_3_val + level2_\(r)[child + 1] * sin_3_val;"
            )
        }
        lines += [
            "        }",
            "        threadgroup_barrier(mem_flags::mem_threadgroup);",
            "",
            "        for (int pair_idx = lane; pair_idx < BlockCount; pair_idx += 32) {",
            "            int bit_offset_4 = pair_idx * \(bits4);",
            "            int word_idx_4 = bit_offset_4 / 32;",
            "            int offset_4 = bit_offset_4 % 32;",
            "            uint value_4 = angle4_ptr[word_idx_4] >> offset_4;",
            "            int spill_4 = offset_4 + \(bits4) - 32;",
            "            if (spill_4 > 0) {",
            "                value_4 |= angle4_ptr[word_idx_4 + 1] << (\(bits4) - spill_4);",
            "            }",
            "            value_4 &= \(mask4)u;",
            "            float cos_4_val = static_cast<float>(cos_4[value_4]);",
            "            float sin_4_val = static_cast<float>(sin_4[value_4]);",
            "            int child = pair_idx << 1;",
            "",
        ]
        for r in 0 ..< repeatCount {
            lines.append(
                "            level4_\(r)[pair_idx] = level3_\(r)[child] * cos_4_val + level3_\(r)[child + 1] * sin_4_val;"
            )
        }
        lines += [
            "        }",
            "        threadgroup_barrier(mem_flags::mem_threadgroup);",
            "",
            "        for (int d = lane; d < Dim; d += 32) {",
            "            int sign_word = d / 32;",
            "            int sign_offset = d % 32;",
            "            uint bit = (sign_ptr[sign_word] >> sign_offset) & 1u;",
            "            float sign = bit ? 1.0f : -1.0f;",
            "",
        ]
        for r in 0 ..< repeatCount {
            lines.append(
                "            qjl_acc_\(r) += static_cast<float>(q_proj_base[\(r) * Dim + d]) * sign;"
            )
        }
        lines += [
            "        }",
            "",
        ]
        for r in 0 ..< repeatCount {
            lines.append(
                "        float polar_acc_\(r) = lane < BlockCount ? static_cast<float>(radii_ptr[lane]) * level4_\(r)[lane] : 0.0f;"
            )
            lines.append("        float polar_sum_\(r) = simd_sum(polar_acc_\(r));")
            lines.append("        float qjl_sum_\(r) = simd_sum(qjl_acc_\(r));")
        }
        lines += [
            "",
            "        if (thread_index_in_simdgroup == 0) {",
        ]
        for r in 0 ..< repeatCount {
            lines.append(
                "            out[((b * kv_heads + h) * RepeatCount + \(r)) * token_count + t] ="
            )
            lines.append(
                "                norm * (polar_sum_\(r) + scale[0] * residual_norm * qjl_sum_\(r));"
            )
        }
        lines += [
            "        }",
        ]
        return MLXFast.metalKernel(
            name: key,
            inputNames: inputNames,
            outputNames: ["out"],
            source: lines.joined(separator: "\n")
        )
    }
}

private func _mseWeightedRotRepeatKernel(_ repeatCount: Int) -> MLXFast.MLXFastKernel? {
    guard _metalAvailable(), repeatCount > 1 else { return nil }
    let key = "turboquant_mse_weighted_rot_repeat_\(repeatCount)"
    return _cachedKernel(key: key) {
        var lines = [
            "        auto lane = thread_position_in_grid.x;",
            "        auto dim_idx = thread_position_in_grid.y;",
            "        auto n = thread_position_in_grid.z;",
            "",
            "        if (dim_idx >= Dim) {",
            "            return;",
            "        }",
            "",
            "        auto token_count = norms_shape[2];",
            "        auto kv_heads = norms_shape[1];",
            "        auto repeat_count = weights_shape[2];",
            "        auto b = n / kv_heads;",
            "        auto h = n % kv_heads;",
            "",
            "        auto weights_base = weights + ((b * kv_heads + h) * repeat_count) * token_count;",
            "        auto norms_ptr = norms + (b * kv_heads + h) * token_count;",
            "        auto packed_ptr = packed + ((b * kv_heads + h) * token_count) * PackedWidth;",
            "",
            "        int bit_offset = dim_idx * Bits;",
            "        int word_idx = bit_offset / 32;",
            "        int offset = bit_offset % 32;",
            "",
        ]
        for r in 0 ..< repeatCount {
            lines.append("        float acc_\(r) = 0.0f;")
        }
        lines += [
            "",
            "        for (int t = lane; t < token_count; t += 32) {",
            "            auto token_ptr = packed_ptr + t * PackedWidth;",
            "            uint value = token_ptr[word_idx] >> offset;",
            "            int spill = offset + Bits - 32;",
            "            if (spill > 0) {",
            "                value |= token_ptr[word_idx + 1] << (Bits - spill);",
            "            }",
            "            value &= ((1u << Bits) - 1u);",
            "            float code = codebook[value];",
            "            float norm = static_cast<float>(norms_ptr[t]);",
        ]
        for r in 0 ..< repeatCount {
            lines.append(
                "            acc_\(r) += static_cast<float>(weights_base[\(r) * token_count + t]) * norm * code;"
            )
        }
        lines += [
            "        }",
            "",
        ]
        for r in 0 ..< repeatCount {
            lines.append("        float acc_sum_\(r) = simd_sum(acc_\(r));")
        }
        lines += [
            "",
            "        if (thread_index_in_simdgroup == 0) {",
        ]
        for r in 0 ..< repeatCount {
            lines.append(
                "            out[((b * kv_heads + h) * repeat_count + \(r)) * Dim + dim_idx] = acc_sum_\(r);"
            )
        }
        lines += ["        }"]
        return MLXFast.metalKernel(
            name: key,
            inputNames: ["weights", "norms", "packed", "codebook"],
            outputNames: ["out"],
            source: lines.joined(separator: "\n")
        )
    }
}

private func _mseScoresWeightedRotRepeatKernel(_ repeatCount: Int) -> MLXFast.MLXFastKernel? {
    guard _metalAvailable(), repeatCount > 1 else { return nil }
    let key = "turboquant_mse_scores_weighted_rot_repeat_\(repeatCount)"
    return _cachedKernel(key: key) {
        var lines = [
            "        auto lane = thread_position_in_grid.x;",
            "        auto dim_idx = thread_position_in_grid.y;",
            "        auto n = thread_position_in_grid.z;",
            "",
            "        if (dim_idx >= Dim) {",
            "            return;",
            "        }",
            "",
            "        auto token_count = norms_shape[2];",
            "        auto kv_heads = norms_shape[1];",
            "        auto repeat_count = scores_shape[2];",
            "        auto b = n / kv_heads;",
            "        auto h = n % kv_heads;",
            "",
            "        auto scores_base = scores + ((b * kv_heads + h) * repeat_count) * token_count;",
            "        auto norms_ptr = norms + (b * kv_heads + h) * token_count;",
            "        auto packed_ptr = packed + ((b * kv_heads + h) * token_count) * PackedWidth;",
            "",
            "        int bit_offset = dim_idx * Bits;",
            "        int word_idx = bit_offset / 32;",
            "        int offset = bit_offset % 32;",
            "",
        ]
        for r in 0 ..< repeatCount { lines.append("        float max_\(r) = -INFINITY;") }
        lines += [
            "",
            "        for (int t = lane; t < token_count; t += 32) {",
        ]
        for r in 0 ..< repeatCount {
            lines.append(
                "            max_\(r) = max(max_\(r), static_cast<float>(scores_base[\(r) * token_count + t]));"
            )
        }
        lines += ["        }", ""]
        for r in 0 ..< repeatCount {
            lines.append("        float max_score_\(r) = simd_max(max_\(r));")
        }
        lines.append("")
        for r in 0 ..< repeatCount {
            lines.append("        float acc_\(r) = 0.0f;")
            lines.append("        float denom_\(r) = 0.0f;")
        }
        lines += [
            "",
            "        for (int t = lane; t < token_count; t += 32) {",
            "            auto token_ptr = packed_ptr + t * PackedWidth;",
            "            uint value = token_ptr[word_idx] >> offset;",
            "            int spill = offset + Bits - 32;",
            "            if (spill > 0) {",
            "                value |= token_ptr[word_idx + 1] << (Bits - spill);",
            "            }",
            "            value &= ((1u << Bits) - 1u);",
            "            float code = codebook[value];",
            "            float norm = static_cast<float>(norms_ptr[t]);",
        ]
        for r in 0 ..< repeatCount {
            lines.append(
                "            float weight_\(r) = exp(static_cast<float>(scores_base[\(r) * token_count + t]) - max_score_\(r));"
            )
            lines.append("            acc_\(r) += weight_\(r) * norm * code;")
            lines.append("            denom_\(r) += weight_\(r);")
        }
        lines += ["        }", ""]
        for r in 0 ..< repeatCount {
            lines.append("        float acc_sum_\(r) = simd_sum(acc_\(r));")
            lines.append("        float denom_sum_\(r) = simd_sum(denom_\(r));")
        }
        lines += ["", "        if (thread_index_in_simdgroup == 0) {"]
        for r in 0 ..< repeatCount {
            lines.append(
                "            out[((b * kv_heads + h) * repeat_count + \(r)) * Dim + dim_idx] ="
            )
            lines.append("                acc_sum_\(r) / max(denom_sum_\(r), 1e-6f);")
        }
        lines += ["        }"]
        return MLXFast.metalKernel(
            name: key,
            inputNames: ["scores", "norms", "packed", "codebook"],
            outputNames: ["out"],
            source: lines.joined(separator: "\n")
        )
    }
}

private func _mseScoresWeightedRotSumRepeatKernel(_ repeatCount: Int) -> MLXFast.MLXFastKernel? {
    guard _metalAvailable(), repeatCount > 1 else { return nil }
    let key = "turboquant_mse_scores_weighted_rot_sum_repeat_\(repeatCount)"
    return _cachedKernel(key: key) {
        var lines = [
            "        auto lane = thread_position_in_grid.x;",
            "        auto dim_idx = thread_position_in_grid.y;",
            "        auto n = thread_position_in_grid.z;",
            "",
            "        if (dim_idx >= Dim) {",
            "            return;",
            "        }",
            "",
            "        auto token_count = norms_shape[2];",
            "        auto kv_heads = norms_shape[1];",
            "        auto repeat_count = scores_shape[2];",
            "        auto b = n / kv_heads;",
            "        auto h = n % kv_heads;",
            "",
            "        auto scores_base = scores + ((b * kv_heads + h) * repeat_count) * token_count;",
            "        auto norms_ptr = norms + (b * kv_heads + h) * token_count;",
            "        auto packed_ptr = packed + ((b * kv_heads + h) * token_count) * PackedWidth;",
            "",
            "        int bit_offset = dim_idx * Bits;",
            "        int word_idx = bit_offset / 32;",
            "        int offset = bit_offset % 32;",
            "",
        ]
        for r in 0 ..< repeatCount { lines.append("        float max_\(r) = -INFINITY;") }
        lines += [
            "",
            "        for (int t = lane; t < token_count; t += 32) {",
        ]
        for r in 0 ..< repeatCount {
            lines.append(
                "            max_\(r) = max(max_\(r), static_cast<float>(scores_base[\(r) * token_count + t]));"
            )
        }
        lines += ["        }", ""]
        for r in 0 ..< repeatCount {
            lines.append("        float max_score_\(r) = simd_max(max_\(r));")
            lines.append("        float acc_\(r) = 0.0f;")
        }
        lines += [
            "",
            "        for (int t = lane; t < token_count; t += 32) {",
            "            auto token_ptr = packed_ptr + t * PackedWidth;",
            "            uint value = token_ptr[word_idx] >> offset;",
            "            int spill = offset + Bits - 32;",
            "            if (spill > 0) {",
            "                value |= token_ptr[word_idx + 1] << (Bits - spill);",
            "            }",
            "            value &= ((1u << Bits) - 1u);",
            "            float code = codebook[value];",
            "            float norm = static_cast<float>(norms_ptr[t]);",
        ]
        for r in 0 ..< repeatCount {
            lines.append(
                "            float weight_\(r) = exp(static_cast<float>(scores_base[\(r) * token_count + t]) - max_score_\(r));"
            )
            lines.append("            acc_\(r) += weight_\(r) * norm * code;")
        }
        lines += ["        }", ""]
        for r in 0 ..< repeatCount {
            lines.append("        float acc_sum_\(r) = simd_sum(acc_\(r));")
        }
        lines += ["", "        if (thread_index_in_simdgroup == 0) {"]
        for r in 0 ..< repeatCount {
            lines.append(
                "            out[((b * kv_heads + h) * repeat_count + \(r)) * Dim + dim_idx] = acc_sum_\(r);"
            )
        }
        lines += ["        }"]
        return MLXFast.metalKernel(
            name: key,
            inputNames: ["scores", "norms", "packed", "codebook"],
            outputNames: ["out"],
            source: lines.joined(separator: "\n")
        )
    }
}

func _validateTurboBits(_ bits: Double) throws -> Double {
    guard bits >= 1 else {
        throw NSError(domain: "TurboQuant", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "TurboQuant requires kv_bits >= 1."
        ])
    }
    let rounded = (bits * 2).rounded() / 2
    guard abs(bits - rounded) <= 1e-6 else {
        throw NSError(domain: "TurboQuant", code: 2, userInfo: [
            NSLocalizedDescriptionKey:
                "TurboQuant currently supports integer and .5 bit-widths, got \(bits)."
        ])
    }
    return rounded
}

public func turboQuantEnabled(bits: Double?, scheme: String? = nil) -> Bool {
    guard let bits else { return false }
    if scheme == "turboquant" {
        return true
    }
    return abs(bits - bits.rounded()) > 1e-6
}

private func _isPowerOfTwo(_ value: Int) -> Bool {
    value > 0 && (value & (value - 1)) == 0
}

private func _polarLevels(_ dim: Int) -> Int {
    guard dim > 1 else { return 0 }
    return min(_polarMaxLevels, Int(log2(Double(dim))))
}

private func _polarLevelBits(_ dim: Int, _ bits: Int) throws -> [Int] {
    guard bits == 4 else {
        throw NSError(domain: "TurboQuant", code: 3, userInfo: [
            NSLocalizedDescriptionKey:
                "PolarQuant key codec currently expects 4 bits, got \(bits)."
        ])
    }
    let levels = _polarLevels(dim)
    guard levels > 0 else { return [] }
    return [4] + Array(repeating: 2, count: max(0, levels - 1))
}

private func _cachedArray(
    store: inout [String: MLXArray], key: String, build: () -> MLXArray
) -> MLXArray {
    _TurboQuantRegistry.lock.lock()
    defer { _TurboQuantRegistry.lock.unlock() }
    if let value = store[key] {
        return value
    }
    let value = build()
    store[key] = value
    return value
}

private func _rotationMatrix(dim: Int, seed: Int) -> MLXArray {
    let key = "\(dim):\(seed)"
    return _cachedArray(store: &_TurboQuantRegistry.rotations, key: key) {
        if dim <= 0 {
            return MLXArray.zeros([0, 0], dtype: .float32)
        }
        if dim == 1 {
            return MLXArray.ones([1, 1], dtype: .float32)
        }
        let state = MLXRandom.RandomState(seed: UInt64(seed + dim * 7919))
        return Stream.withNewDefaultStream(device: .cpu) {
            let matrix = withRandomState(state) {
                normal([dim, dim], dtype: .float32)
            }
            let (q, r) = qr(matrix)
            return q * sign(diag(r))
        }
    }
}

private func _projectionMatrix(dim: Int, seed: Int) -> MLXArray {
    let key = "\(dim):\(seed)"
    return _cachedArray(store: &_TurboQuantRegistry.projections, key: key) {
        if dim <= 0 {
            return MLXArray.zeros([0, 0], dtype: .float32)
        }
        let state = MLXRandom.RandomState(seed: UInt64(seed + dim * 2971 + 17))
        return Stream.withNewDefaultStream(device: .cpu) {
            withRandomState(state) {
                normal([dim, dim], dtype: .float32)
            }
        }
    }
}

private func _betaPDF(_ grid: [Float], dim: Int) -> [Float] {
    let weights: [Double]
    if dim <= 1 {
        weights = Array(repeating: 1.0, count: grid.count)
    } else {
        let coeff = tgamma(Double(dim) / 2)
            / (sqrt(Double.pi) * tgamma((Double(dim) - 1) / 2))
        weights = grid.map { value in
            coeff * Foundation.pow(max(1.0 - Double(value * value), 0.0), Double(dim - 3) / 2)
        }
    }
    let total = weights.reduce(0, +)
    guard total > 0 else {
        return Array(repeating: 1.0 / Float(grid.count), count: grid.count)
    }
    return weights.map { Float($0 / total) }
}

private func _interp(_ x: Float, cdf: [Float], grid: [Float]) -> Float {
    if x <= cdf[0] {
        return grid[0]
    }
    if x >= cdf[cdf.count - 1] {
        return grid[grid.count - 1]
    }

    var lo = 0
    var hi = cdf.count - 1
    while lo + 1 < hi {
        let mid = (lo + hi) / 2
        if cdf[mid] < x {
            lo = mid
        } else {
            hi = mid
        }
    }

    let c0 = cdf[lo]
    let c1 = cdf[hi]
    let g0 = grid[lo]
    let g1 = grid[hi]
    let denom = max(c1 - c0, 1e-12)
    let alpha = (x - c0) / denom
    return g0 + alpha * (g1 - g0)
}

private func _codebook(_ dim: Int, _ bits: Int) -> MLXArray {
    let key = "\(dim):\(bits)"
    return _cachedArray(store: &_TurboQuantRegistry.codebooks, key: key) {
        if bits <= 0 {
            return MLXArray.zeros([0], dtype: .float32)
        }

        let levels = 1 << bits
        if dim <= 1 {
            return linspace(-1.0 as Float, 1.0 as Float, count: levels)
        }

        let grid = (0 ..< 32_768).map { index -> Float in
            let alpha = Float(index) / Float(32_767)
            return (-1.0 + 1e-6) + alpha * (2.0 - 2e-6)
        }
        let weights = _betaPDF(grid, dim: dim)

        var cdf: [Float] = []
        cdf.reserveCapacity(weights.count)
        var running: Float = 0
        for weight in weights {
            running += weight
            cdf.append(running)
        }

        var centroids = (0 ..< levels).map { level -> Float in
            let quantile = (Float(level) + 0.5) / Float(levels)
            return _interp(quantile, cdf: cdf, grid: grid)
        }

        for _ in 0 ..< 100 {
            var boundaries = Array(repeating: Float.zero, count: levels + 1)
            boundaries[0] = -1.0
            boundaries[levels] = 1.0
            if levels > 1 {
                for index in 1 ..< levels {
                    boundaries[index] = 0.5 * (centroids[index - 1] + centroids[index])
                }
            }

            var next = centroids
            for index in 0 ..< levels {
                var weightedSum: Double = 0
                var totalWeight: Double = 0
                for position in 0 ..< grid.count {
                    let value = grid[position]
                    let inBucket =
                        if index == levels - 1 {
                            value >= boundaries[index] && value <= boundaries[index + 1]
                        } else {
                            value >= boundaries[index] && value < boundaries[index + 1]
                        }
                    if inBucket {
                        let weight = Double(weights[position])
                        weightedSum += weight * Double(value)
                        totalWeight += weight
                    }
                }
                if totalWeight > 0 {
                    next[index] = Float(weightedSum / totalWeight)
                }
            }

            let delta = zip(next, centroids).map { abs($0 - $1) }.max() ?? 0
            centroids = next
            if delta < 1e-6 {
                break
            }
        }

        return MLXArray(centroids)
    }
}

private func _polarAnglePDF(_ grid: [Float], level: Int) -> [Float] {
    let weights: [Double]
    if level <= 1 {
        weights = Array(repeating: 1.0, count: grid.count)
    } else {
        let exponent = (1 << (level - 1)) - 1
        weights = grid.map { value in
            Foundation.pow(max(Foundation.sin(2.0 * Double(value)), 0.0), Double(exponent))
        }
    }
    let total = weights.reduce(0, +)
    guard total > 0 else {
        return Array(repeating: 1.0 / Float(grid.count), count: grid.count)
    }
    return weights.map { Float($0 / total) }
}

private func _polarAngleCodebook(_ level: Int, _ bits: Int) -> MLXArray {
    let key = "\(level):\(bits)"
    return _cachedArray(store: &_TurboQuantRegistry.polarCodebooks, key: key) {
        if bits <= 0 {
            return MLXArray.zeros([0], dtype: .float32)
        }

        let levelCount = 1 << bits
        if level <= 1 {
            let step = Float(2.0 * Double.pi) / Float(levelCount)
            let centroids = (0 ..< levelCount).map { Float($0) * step + step / 2 }
            return MLXArray(centroids)
        }

        let grid = (0 ..< 32_768).map { index -> Float in
            let alpha = Float(index) / Float(32_767)
            return 1e-6 + alpha * (Float.pi / 2 - 2e-6)
        }
        let weights = _polarAnglePDF(grid, level: level)
        var cdf: [Float] = []
        var running: Float = 0
        cdf.reserveCapacity(grid.count)
        for weight in weights {
            running += weight
            cdf.append(running)
        }

        var centroids = (0 ..< levelCount).map { index -> Float in
            let quantile = (Float(index) + 0.5) / Float(levelCount)
            return _interp(quantile, cdf: cdf, grid: grid)
        }

        for _ in 0 ..< 100 {
            var boundaries = Array(repeating: Float.zero, count: levelCount + 1)
            boundaries[0] = 0
            boundaries[levelCount] = Float.pi / 2
            if levelCount > 1 {
                for index in 1 ..< levelCount {
                    boundaries[index] = 0.5 * (centroids[index - 1] + centroids[index])
                }
            }

            var next = centroids
            for index in 0 ..< levelCount {
                var weightedSum: Double = 0
                var totalWeight: Double = 0
                for position in 0 ..< grid.count {
                    let value = grid[position]
                    let inBucket =
                        if index == levelCount - 1 {
                            value >= boundaries[index] && value <= boundaries[index + 1]
                        } else {
                            value >= boundaries[index] && value < boundaries[index + 1]
                        }
                    if inBucket {
                        let weight = Double(weights[position])
                        weightedSum += weight * Double(value)
                        totalWeight += weight
                    }
                }
                if totalWeight > 0 {
                    next[index] = Float(weightedSum / totalWeight)
                }
            }

            let delta = zip(next, centroids).map { abs($0 - $1) }.max() ?? 0
            centroids = next
            if delta < 1e-6 {
                break
            }
        }

        return MLXArray(centroids)
    }
}

private func _packedWidth(length: Int, bits: Int) -> Int {
    guard length > 0, bits > 0 else { return 0 }
    return (length * bits + 31) / 32
}

private func _packLowbit(_ values: MLXArray, bits: Int) -> MLXArray {
    if bits == 0 {
        return MLXArray.zeros(Array(values.shape.dropLast()) + [0], dtype: .uint32)
    }

    let values = values.asType(.uint32)
    let length = values.dim(-1)
    let packedWidth = _packedWidth(length: length, bits: bits)
    let flat = values.reshaped(-1, length)

    if let kernel = _packLowbitKernel() {
        let packed = kernel(
            [flat],
            template: [
                ("Bits", bits),
                ("Length", length),
                ("PackedWidth", packedWidth),
            ],
            grid: (packedWidth, flat.dim(0), 1),
            threadGroup: (min(32, packedWidth), 1, 1),
            outputShapes: [[flat.dim(0), packedWidth]],
            outputDTypes: [.uint32]
        )[0]
        return packed.reshaped(Array(values.shape.dropLast()) + [packedWidth])
    }

    let rows = flat.dim(0)
    let source = flat.asArray(UInt32.self)
    var packed = Array(repeating: UInt32.zero, count: rows * packedWidth)
    for row in 0 ..< rows {
        for index in 0 ..< length {
            let value = source[row * length + index]
            let bitOffset = index * bits
            let wordIndex = bitOffset / 32
            let offset = bitOffset % 32
            let base = row * packedWidth + wordIndex
            packed[base] |= value << UInt32(offset)
            let spill = offset + bits - 32
            if spill > 0 {
                packed[base + 1] |= value >> UInt32(bits - spill)
            }
        }
    }
    return MLXArray(packed).reshaped(Array(values.shape.dropLast()) + [packedWidth])
}

private func _unpackLowbit(_ packed: MLXArray, bits: Int, length: Int) -> MLXArray {
    if bits == 0 {
        return MLXArray.zeros(Array(packed.shape.dropLast()) + [0], dtype: .uint32)
    }

    let packed = packed.asType(.uint32)
    let flat = packed.reshaped(-1, packed.dim(-1))
    if let kernel = _unpackLowbitKernel() {
        let unpacked = kernel(
            [flat],
            template: [
                ("Bits", bits),
                ("Length", length),
                ("PackedWidth", flat.dim(-1)),
            ],
            grid: (length, flat.dim(0), 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[flat.dim(0), length]],
            outputDTypes: [.uint32]
        )[0]
        return unpacked.reshaped(Array(packed.shape.dropLast()) + [length])
    }

    let rows = flat.dim(0)
    let packedWidth = flat.dim(-1)
    let source = flat.asArray(UInt32.self)
    var unpacked = Array(repeating: UInt32.zero, count: rows * length)
    let mask = UInt32((1 << bits) - 1)
    for row in 0 ..< rows {
        for index in 0 ..< length {
            let bitOffset = index * bits
            let wordIndex = bitOffset / 32
            let offset = bitOffset % 32
            let base = row * packedWidth + wordIndex
            var value = source[base] >> UInt32(offset)
            let spill = offset + bits - 32
            if spill > 0 {
                value |= source[base + 1] << UInt32(bits - spill)
            }
            unpacked[row * length + index] = value & mask
        }
    }
    return MLXArray(unpacked).reshaped(Array(packed.shape.dropLast()) + [length])
}

private func _stateLength(_ state: TurboQuantState?) -> Int {
    guard let state else { return 0 }
    switch state {
    case .mse(let state):
        return state.norms.dim(2)
    case .prod(let state):
        return state.norms.dim(2)
    case .polar(let state):
        return state.radii.dim(2)
    case .polarProd(let state):
        return state.norms.dim(2)
    case .split(let state):
        return _stateLength(state.low)
    }
}

private func _stateNBytes(_ state: TurboQuantState?) -> Int {
    guard let state else { return 0 }
    switch state {
    case .mse(let state):
        return state.norms.nbytes + state.indices.nbytes
    case .prod(let state):
        return state.norms.nbytes + state.mseIndices.nbytes + state.residualNorms.nbytes
            + state.qjlSigns.nbytes
    case .polar(let state):
        return state.levelIndices.reduce(state.radii.nbytes) { $0 + $1.nbytes }
    case .polarProd(let state):
        return state.norms.nbytes + _stateNBytes(.polar(state.polarState)) + state.residualNorms.nbytes
            + state.qjlSigns.nbytes
    case .split(let state):
        return _stateNBytes(state.low) + _stateNBytes(state.high)
    }
}

private func _flattenState(_ state: TurboQuantState) -> [MLXArray] {
    switch state {
    case .mse(let state):
        return [state.norms, state.indices]
    case .prod(let state):
        return [state.norms, state.mseIndices, state.residualNorms, state.qjlSigns]
    case .polar(let state):
        return [state.radii] + state.levelIndices
    case .polarProd(let state):
        return [state.norms] + _flattenState(.polar(state.polarState)) + [state.residualNorms, state.qjlSigns]
    case .split(let state):
        return _flattenState(state.low) + _flattenState(state.high)
    }
}

private func _sliceState(_ state: TurboQuantState?, end: Int) -> TurboQuantState? {
    guard let state else { return nil }
    switch state {
    case .mse(let state):
        return .mse(TurboQuantMSEState(
            norms: state.norms[.ellipsis, ..<end],
            indices: state.indices[.ellipsis, ..<end, 0...]
        ))
    case .prod(let state):
        return .prod(TurboQuantProdState(
            norms: state.norms[.ellipsis, ..<end],
            mseIndices: state.mseIndices[.ellipsis, ..<end, 0...],
            residualNorms: state.residualNorms[.ellipsis, ..<end],
            qjlSigns: state.qjlSigns[.ellipsis, ..<end, 0...]
        ))
    case .polar(let state):
        return .polar(TurboQuantPolarState(
            radii: state.radii[.ellipsis, ..<end, 0...],
            levelIndices: state.levelIndices.map { $0[.ellipsis, ..<end, 0...] }
        ))
    case .polarProd(let state):
        return .polarProd(TurboQuantPolarProdState(
            norms: state.norms[.ellipsis, ..<end],
            polarState: {
                guard case .polar(let polarState) = _sliceState(.polar(state.polarState), end: end)! else {
                    fatalError("Invalid TurboQuant polar state")
                }
                return polarState
            }(),
            residualNorms: state.residualNorms[.ellipsis, ..<end],
            qjlSigns: state.qjlSigns[.ellipsis, ..<end, 0...]
        ))
    case .split(let state):
        return .split(TurboQuantSplitState(
            low: _sliceState(state.low, end: end)!,
            high: _sliceState(state.high, end: end)!
        ))
    }
}

private func _sliceStateRange(_ state: TurboQuantState?, start: Int, end: Int) -> TurboQuantState? {
    guard let state else { return nil }
    switch state {
    case .mse(let state):
        return .mse(TurboQuantMSEState(
            norms: state.norms[.ellipsis, start ..< end],
            indices: state.indices[.ellipsis, start ..< end, 0...]
        ))
    case .prod(let state):
        return .prod(TurboQuantProdState(
            norms: state.norms[.ellipsis, start ..< end],
            mseIndices: state.mseIndices[.ellipsis, start ..< end, 0...],
            residualNorms: state.residualNorms[.ellipsis, start ..< end],
            qjlSigns: state.qjlSigns[.ellipsis, start ..< end, 0...]
        ))
    case .polar(let state):
        return .polar(TurboQuantPolarState(
            radii: state.radii[.ellipsis, start ..< end, 0...],
            levelIndices: state.levelIndices.map { $0[.ellipsis, start ..< end, 0...] }
        ))
    case .polarProd(let state):
        return .polarProd(TurboQuantPolarProdState(
            norms: state.norms[.ellipsis, start ..< end],
            polarState: {
                guard case .polar(let polarState) = _sliceStateRange(.polar(state.polarState), start: start, end: end)! else {
                    fatalError("Invalid TurboQuant polar state")
                }
                return polarState
            }(),
            residualNorms: state.residualNorms[.ellipsis, start ..< end],
            qjlSigns: state.qjlSigns[.ellipsis, start ..< end, 0...]
        ))
    case .split(let state):
        return .split(TurboQuantSplitState(
            low: _sliceStateRange(state.low, start: start, end: end)!,
            high: _sliceStateRange(state.high, start: start, end: end)!
        ))
    }
}

private func _allocateStateLike(_ state: TurboQuantState, length: Int) -> TurboQuantState {
    switch state {
    case .mse(let state):
        return .mse(TurboQuantMSEState(
            norms: MLXArray.zeros(Array(state.norms.shape.prefix(2)) + [length], dtype: state.norms.dtype),
            indices: MLXArray.zeros(Array(state.indices.shape.prefix(2)) + [length, state.indices.dim(-1)], dtype: state.indices.dtype)
        ))
    case .prod(let state):
        return .prod(TurboQuantProdState(
            norms: MLXArray.zeros(Array(state.norms.shape.prefix(2)) + [length], dtype: state.norms.dtype),
            mseIndices: MLXArray.zeros(Array(state.mseIndices.shape.prefix(2)) + [length, state.mseIndices.dim(-1)], dtype: state.mseIndices.dtype),
            residualNorms: MLXArray.zeros(Array(state.residualNorms.shape.prefix(2)) + [length], dtype: state.residualNorms.dtype),
            qjlSigns: MLXArray.zeros(Array(state.qjlSigns.shape.prefix(2)) + [length, state.qjlSigns.dim(-1)], dtype: state.qjlSigns.dtype)
        ))
    case .polar(let state):
        return .polar(TurboQuantPolarState(
            radii: MLXArray.zeros(Array(state.radii.shape.prefix(2)) + [length, state.radii.dim(-1)], dtype: state.radii.dtype),
            levelIndices: state.levelIndices.map {
                MLXArray.zeros(Array($0.shape.prefix(2)) + [length, $0.dim(-1)], dtype: $0.dtype)
            }
        ))
    case .polarProd(let state):
        return .polarProd(TurboQuantPolarProdState(
            norms: MLXArray.zeros(Array(state.norms.shape.prefix(2)) + [length], dtype: state.norms.dtype),
            polarState: {
                guard case .polar(let polarState) = _allocateStateLike(.polar(state.polarState), length: length) else {
                    fatalError("Invalid TurboQuant polar state")
                }
                return polarState
            }(),
            residualNorms: MLXArray.zeros(Array(state.residualNorms.shape.prefix(2)) + [length], dtype: state.residualNorms.dtype),
            qjlSigns: MLXArray.zeros(Array(state.qjlSigns.shape.prefix(2)) + [length, state.qjlSigns.dim(-1)], dtype: state.qjlSigns.dtype)
        ))
    case .split(let state):
        return .split(TurboQuantSplitState(
            low: _allocateStateLike(state.low, length: length),
            high: _allocateStateLike(state.high, length: length)
        ))
    }
}

private func _writeState(_ destination: inout TurboQuantState, source: TurboQuantState, start: Int) {
    let end = start + _stateLength(source)
    switch (destination, source) {
    case (.mse(var destinationState), .mse(let sourceState)):
        destinationState.norms[.ellipsis, start ..< end] = sourceState.norms
        destinationState.indices[.ellipsis, start ..< end, 0...] = sourceState.indices
        destination = .mse(destinationState)
    case (.prod(var destinationState), .prod(let sourceState)):
        destinationState.norms[.ellipsis, start ..< end] = sourceState.norms
        destinationState.mseIndices[.ellipsis, start ..< end, 0...] = sourceState.mseIndices
        destinationState.residualNorms[.ellipsis, start ..< end] = sourceState.residualNorms
        destinationState.qjlSigns[.ellipsis, start ..< end, 0...] = sourceState.qjlSigns
        destination = .prod(destinationState)
    case (.polar(var destinationState), .polar(let sourceState)):
        destinationState.radii[.ellipsis, start ..< end, 0...] = sourceState.radii
        for (index, sourceLevel) in sourceState.levelIndices.enumerated() {
            destinationState.levelIndices[index][.ellipsis, start ..< end, 0...] = sourceLevel
        }
        destination = .polar(destinationState)
    case (.polarProd(var destinationState), .polarProd(let sourceState)):
        destinationState.norms[.ellipsis, start ..< end] = sourceState.norms
        var polarState = TurboQuantState.polar(destinationState.polarState)
        _writeState(&polarState, source: .polar(sourceState.polarState), start: start)
        guard case .polar(let updatedPolar) = polarState else {
            fatalError("Invalid TurboQuant polar state")
        }
        destinationState.polarState = updatedPolar
        destinationState.residualNorms[.ellipsis, start ..< end] = sourceState.residualNorms
        destinationState.qjlSigns[.ellipsis, start ..< end, 0...] = sourceState.qjlSigns
        destination = .polarProd(destinationState)
    case (.split(var destinationState), .split(let sourceState)):
        _writeState(&destinationState.low, source: sourceState.low, start: start)
        _writeState(&destinationState.high, source: sourceState.high, start: start)
        destination = .split(destinationState)
    default:
        fatalError("Unsupported TurboQuant state write")
    }
}

private func _reserveStateCapacity(
    _ state: TurboQuantState?, used: Int, needed: Int, step: Int
) -> TurboQuantState? {
    guard let state else { return nil }
    let capacity = _stateLength(state)
    if capacity >= needed {
        return state
    }
    var newCapacity = max(needed, max(capacity * 2, step))
    newCapacity = ((newCapacity + step - 1) / step) * step
    var grown = _allocateStateLike(state, length: newCapacity)
    if used > 0, let sliced = _sliceState(state, end: used) {
        _writeState(&grown, source: sliced, start: 0)
    }
    return grown
}

private func _kvHeadCount(_ state: TurboQuantState) -> Int {
    switch state {
    case .mse(let state):
        return state.norms.dim(1)
    case .prod(let state):
        return state.norms.dim(1)
    case .polar(let state):
        return state.radii.dim(1)
    case .polarProd(let state):
        return state.norms.dim(1)
    case .split(let state):
        return _kvHeadCount(state.low)
    }
}

private func _metalMSEScore(
    qRot: MLXArray,
    state: TurboQuantMSEState,
    bits: Int,
    codebook: MLXArray
) -> MLXArray? {
    guard bits > 0, _metalAvailable(), qRot.ndim == 4, state.norms.dim(2) > 0 else {
        return nil
    }
    guard let kernel = _mseScoreKernel() else { return nil }

    let (b, h, r, d) = (qRot.dim(0), qRot.dim(1), qRot.dim(2), qRot.dim(3))
    let t = state.norms.dim(2)
    let scores = kernel(
        [qRot, state.norms, state.indices.asType(.uint32), codebook],
        template: [
            ("Dim", d),
            ("Bits", bits),
            ("PackedWidth", state.indices.dim(-1)),
        ],
        grid: (32, r, b * h * t),
        threadGroup: (32, 1, 1),
        outputShapes: [[b, h, r, t]],
        outputDTypes: [.float32]
    )[0]
    return expandedDimensions(scores, axis: 3)
}

private func _metalQJLScore(
    qProj: MLXArray,
    state: TurboQuantProdState,
    scale: MLXArray
) -> MLXArray? {
    guard _metalAvailable(), qProj.ndim == 4, state.norms.dim(2) > 0 else {
        return nil
    }
    guard let kernel = _qjlScoreKernel() else { return nil }

    let (b, h, r, d) = (qProj.dim(0), qProj.dim(1), qProj.dim(2), qProj.dim(3))
    let t = state.norms.dim(2)
    let scores = kernel(
        [
            qProj,
            state.norms,
            state.residualNorms,
            state.qjlSigns.asType(.uint32),
            scale,
        ],
        template: [
            ("Dim", d),
            ("PackedWidth", state.qjlSigns.dim(-1)),
        ],
        grid: (32, r, b * h * t),
        threadGroup: (32, 1, 1),
        outputShapes: [[b, h, r, t]],
        outputDTypes: [.float32]
    )[0]
    return expandedDimensions(scores, axis: 3)
}

private func _metalProdScore(
    qRot: MLXArray,
    qProj: MLXArray,
    state: TurboQuantProdState,
    mseBits: Int,
    codebook: MLXArray,
    scale: MLXArray
) -> MLXArray? {
    guard
        mseBits > 0,
        _metalAvailable(),
        qRot.ndim == 4,
        qProj.ndim == 4,
        state.norms.dim(2) > 0
    else { return nil }

    let (b, h, r, d) = (qRot.dim(0), qRot.dim(1), qRot.dim(2), qRot.dim(3))
    let t = state.norms.dim(2)
    if r > 1, let kernel = _prodScoreRepeatKernel(r) {
        let scores = kernel(
            [
                qRot,
                qProj,
                state.norms,
                state.residualNorms,
                state.mseIndices.asType(.uint32),
                state.qjlSigns.asType(.uint32),
                codebook,
                scale,
            ],
            template: [
                ("Dim", d),
                ("RepeatCount", r),
                ("MseBits", mseBits),
                ("MsePackedWidth", state.mseIndices.dim(-1)),
                ("SignPackedWidth", state.qjlSigns.dim(-1)),
            ],
            grid: (32, 1, b * h * t),
            threadGroup: (32, 1, 1),
            outputShapes: [[b, h, r, t]],
            outputDTypes: [.float32]
        )[0]
        return expandedDimensions(scores, axis: 3)
    }

    guard let kernel = _prodScoreKernel() else { return nil }
    let scores = kernel(
        [
            qRot,
            qProj,
            state.norms,
            state.residualNorms,
            state.mseIndices.asType(.uint32),
            state.qjlSigns.asType(.uint32),
            codebook,
            scale,
        ],
        template: [
            ("Dim", d),
            ("MseBits", mseBits),
            ("MsePackedWidth", state.mseIndices.dim(-1)),
            ("SignPackedWidth", state.qjlSigns.dim(-1)),
        ],
        grid: (32, r, b * h * t),
        threadGroup: (32, 1, 1),
        outputShapes: [[b, h, r, t]],
        outputDTypes: [.float32]
    )[0]
    return expandedDimensions(scores, axis: 3)
}

private func _metalPolarProdScore(
    qRot: MLXArray,
    state: TurboQuantPolarProdState,
    levelBits: [Int],
    cosTables: [MLXArray],
    sinTables: [MLXArray]
) -> MLXArray? {
    guard _metalAvailable(), qRot.ndim == 4, state.norms.dim(2) > 0, !levelBits.isEmpty else {
        return nil
    }
    guard let kernel = _polarProdScoreKernel(levelBits) else { return nil }

    let (b, h, r, d) = (qRot.dim(0), qRot.dim(1), qRot.dim(2), qRot.dim(3))
    let t = state.norms.dim(2)
    var inputs: [MLXArray] = [qRot, state.norms, state.polarState.radii]
    inputs.append(contentsOf: state.polarState.levelIndices.map { $0.asType(.uint32) })
    for (cosTable, sinTable) in zip(cosTables, sinTables) {
        inputs.append(cosTable)
        inputs.append(sinTable)
    }

    var template: [(String, any KernelTemplateArg)] = [
        ("Dim", d),
        ("Levels", levelBits.count),
        ("BlockCount", state.polarState.radii.dim(-1)),
    ]
    for (index, levelState) in state.polarState.levelIndices.enumerated() {
        template.append(("PackedWidth\(index + 1)", levelState.dim(-1)))
    }

    let scores = kernel(
        inputs,
        template: template,
        grid: (32, r, b * h * t),
        threadGroup: (32, 1, 1),
        outputShapes: [[b, h, r, t]],
        outputDTypes: [.float32]
    )[0]
    return expandedDimensions(scores, axis: 3)
}

private func _metalPolarTurboScore(
    qRot: MLXArray,
    qProj: MLXArray,
    state: TurboQuantPolarProdState,
    levelBits: [Int],
    cosTables: [MLXArray],
    sinTables: [MLXArray],
    scale: MLXArray
) -> MLXArray? {
    guard
        _metalAvailable(),
        qRot.ndim == 4,
        qProj.ndim == 4,
        qRot.shape == qProj.shape,
        state.norms.dim(2) > 0,
        !levelBits.isEmpty
    else { return nil }
    let (b, h, r, d) = (qRot.dim(0), qRot.dim(1), qRot.dim(2), qRot.dim(3))
    let t = state.norms.dim(2)
    guard let kernel = _polarTurboScoreRepeatKernel(levelBits: levelBits, repeatCount: r) else {
        return nil
    }

    var inputs: [MLXArray] = [qRot, qProj, state.norms, state.polarState.radii]
    inputs.append(contentsOf: state.polarState.levelIndices.map { $0.asType(.uint32) })
    inputs.append(contentsOf: [state.residualNorms, state.qjlSigns.asType(.uint32), scale])
    for (cosTable, sinTable) in zip(cosTables, sinTables) {
        inputs.append(cosTable)
        inputs.append(sinTable)
    }

    var template: [(String, any KernelTemplateArg)] = [
        ("Dim", d),
        ("Levels", levelBits.count),
        ("RepeatCount", r),
        ("BlockCount", state.polarState.radii.dim(-1)),
        ("SignPackedWidth", state.qjlSigns.dim(-1)),
    ]
    for (index, levelState) in state.polarState.levelIndices.enumerated() {
        template.append(("PackedWidth\(index + 1)", levelState.dim(-1)))
    }

    let scores = kernel(
        inputs,
        template: template,
        grid: (32, 1, b * h * t),
        threadGroup: (32, 1, 1),
        outputShapes: [[b, h, r, t]],
        outputDTypes: [.float32]
    )[0]
    return expandedDimensions(scores, axis: 3)
}

private func _metalMSEWeightedSum(
    weights: MLXArray,
    state: TurboQuantMSEState,
    bits: Int,
    codebook: MLXArray,
    rotation: MLXArray
) -> MLXArray? {
    guard
        bits > 0,
        _metalAvailable(),
        weights.ndim == 5,
        weights.dim(-2) == 1,
        state.norms.dim(2) > 0
    else { return nil }

    let weights2D = weights.reshaped(weights.dim(0), weights.dim(1), weights.dim(2), weights.dim(-1))
    let (b, h, r) = (weights2D.dim(0), weights2D.dim(1), weights2D.dim(2))
    let d = rotation.dim(0)
    if r > 1, let kernel = _mseWeightedRotRepeatKernel(r) {
        let weightedRot = kernel(
            [weights2D, state.norms, state.indices.asType(.uint32), codebook],
            template: [
                ("Dim", d),
                ("RepeatCount", r),
                ("Bits", bits),
                ("PackedWidth", state.indices.dim(-1)),
            ],
            grid: (32, d, b * h),
            threadGroup: (32, 1, 1),
            outputShapes: [[b, h, r, d]],
            outputDTypes: [.float32]
        )[0]
        return expandedDimensions(matmul(weightedRot, rotation), axis: 3)
    }

    guard let kernel = _mseWeightedRotKernel() else { return nil }
    let weightedRot = kernel(
        [weights2D, state.norms, state.indices.asType(.uint32), codebook],
        template: [
            ("Dim", d),
            ("Bits", bits),
            ("PackedWidth", state.indices.dim(-1)),
        ],
        grid: (32, d, b * h * r),
        threadGroup: (32, 1, 1),
        outputShapes: [[b, h, r, d]],
        outputDTypes: [.float32]
    )[0]
    return expandedDimensions(matmul(weightedRot, rotation), axis: 3)
}

private func _metalMSEWeightedSumFromScores(
    scores: MLXArray,
    state: TurboQuantMSEState,
    bits: Int,
    codebook: MLXArray,
    rotation: MLXArray
) -> MLXArray? {
    guard
        bits > 0,
        _metalAvailable(),
        scores.ndim == 5,
        scores.dim(-2) == 1,
        state.norms.dim(2) > 0
    else { return nil }

    let scores2D = scores.reshaped(scores.dim(0), scores.dim(1), scores.dim(2), scores.dim(-1))
    let (b, h, r) = (scores2D.dim(0), scores2D.dim(1), scores2D.dim(2))
    guard r > 1, let kernel = _mseScoresWeightedRotRepeatKernel(r) else { return nil }
    let d = rotation.dim(0)
    let weightedRot = kernel(
        [scores2D, state.norms, state.indices.asType(.uint32), codebook],
        template: [
            ("Dim", d),
            ("Bits", bits),
            ("PackedWidth", state.indices.dim(-1)),
        ],
        grid: (32, d, b * h),
        threadGroup: (32, 1, 1),
        outputShapes: [[b, h, r, d]],
        outputDTypes: [.float32]
    )[0]
    return expandedDimensions(matmul(weightedRot, rotation), axis: 3)
}

private func _metalMSEWeightedSumSumFromScores(
    scores: MLXArray,
    state: TurboQuantMSEState,
    bits: Int,
    codebook: MLXArray,
    rotation: MLXArray
) -> MLXArray? {
    guard
        bits > 0,
        _metalAvailable(),
        scores.ndim == 5,
        scores.dim(-2) == 1,
        state.norms.dim(2) > 0
    else { return nil }

    let scores2D = scores.reshaped(scores.dim(0), scores.dim(1), scores.dim(2), scores.dim(-1))
    let (b, h, r) = (scores2D.dim(0), scores2D.dim(1), scores2D.dim(2))
    guard r > 1, let kernel = _mseScoresWeightedRotSumRepeatKernel(r) else { return nil }
    let d = rotation.dim(0)
    let weightedRot = kernel(
        [scores2D, state.norms, state.indices.asType(.uint32), codebook],
        template: [
            ("Dim", d),
            ("Bits", bits),
            ("PackedWidth", state.indices.dim(-1)),
        ],
        grid: (32, d, b * h),
        threadGroup: (32, 1, 1),
        outputShapes: [[b, h, r, d]],
        outputDTypes: [.float32]
    )[0]
    return expandedDimensions(matmul(weightedRot, rotation), axis: 3)
}

private func _compiledIntegerDecodeKernel(bits: Int) -> @Sendable ([MLXArray]) -> [MLXArray] {
    _TurboQuantRegistry.lock.lock()
    if let decoder = _TurboQuantRegistry.compiledIntegerDecoders[bits] {
        _TurboQuantRegistry.lock.unlock()
        return decoder
    }
    _TurboQuantRegistry.lock.unlock()

    let mseBits = max(bits - 1, 0)
    let decoder = compile { arrays in
        let groupedQueries = arrays[0]
        let keyNorms = arrays[1]
        let keyMSEIndices = arrays[2]
        let keyResidualNorms = arrays[3]
        let keyQJLSigns = arrays[4]
        let valueNorms = arrays[5]
        let valueIndices = arrays[6]
        let keyQueryTransformT = arrays[7]
        let keyCodebook = arrays[8]
        let keyScale = arrays[9]
        let valueCodebook = arrays[10]
        let valueRotation = arrays[11]

        let queryTransformed = matmul(groupedQueries, keyQueryTransformT)
        let dim = groupedQueries.dim(-1)
        let qRot = queryTransformed[.ellipsis, ..<dim].reshaped(
            queryTransformed.dim(0), queryTransformed.dim(1), queryTransformed.dim(2), dim)
        let qProj = queryTransformed[.ellipsis, dim...].reshaped(
            queryTransformed.dim(0), queryTransformed.dim(1), queryTransformed.dim(2), dim)
        let scores = _metalProdScore(
            qRot: qRot,
            qProj: qProj,
            state: TurboQuantProdState(
                norms: keyNorms,
                mseIndices: keyMSEIndices,
                residualNorms: keyResidualNorms,
                qjlSigns: keyQJLSigns
            ),
            mseBits: mseBits,
            codebook: keyCodebook,
            scale: keyScale
        )!
        let output = _metalMSEWeightedSumFromScores(
            scores: scores,
            state: TurboQuantMSEState(norms: valueNorms, indices: valueIndices),
            bits: bits,
            codebook: valueCodebook,
            rotation: valueRotation
        )!
        return [output]
    }

    _TurboQuantRegistry.lock.lock()
    _TurboQuantRegistry.compiledIntegerDecoders[bits] = decoder
    _TurboQuantRegistry.lock.unlock()
    return decoder
}

protocol _TurboQuantCodec: AnyObject {
    var dim: Int { get }
    var descriptor: String { get }
    var serializationArrays: [MLXArray] { get }

    func quantize(_ vectors: MLXArray) -> TurboQuantState
    func dequantize(_ state: TurboQuantState) -> MLXArray
    func prepareQueries(_ queries: MLXArray) -> TurboQuantPreparedQueries
    func scorePrepared(_ preparedQueries: TurboQuantPreparedQueries, state: TurboQuantState)
        -> MLXArray
    func weightedSum(_ weights: MLXArray, state: TurboQuantState) -> MLXArray
    func weightedSumFromScores(_ scores: MLXArray, state: TurboQuantState) -> MLXArray
    func weightedSumStatsFromScores(_ scores: MLXArray, state: TurboQuantState)
        -> (MLXArray, MLXArray, MLXArray)
}

extension _TurboQuantCodec {
    var serializationArrays: [MLXArray] { [] }

    func weightedSum(_ weights: MLXArray, state: TurboQuantState) -> MLXArray {
        fatalError("weightedSum is unsupported for this TurboQuant codec")
    }

    func weightedSumFromScores(_ scores: MLXArray, state: TurboQuantState) -> MLXArray {
        fatalError("weightedSumFromScores is unsupported for this TurboQuant codec")
    }

    func weightedSumStatsFromScores(_ scores: MLXArray, state: TurboQuantState)
        -> (MLXArray, MLXArray, MLXArray)
    {
        fatalError("weightedSumStatsFromScores is unsupported for this TurboQuant codec")
    }
}

final class _TurboQuantMSECodec: _TurboQuantCodec {
    let dim: Int
    let bits: Int
    let seed: Int
    let rotation: MLXArray
    let rotationT: MLXArray
    let codebook: MLXArray

    init(_ dim: Int, _ bits: Int, seed: Int) {
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.rotation = _rotationMatrix(dim: dim, seed: seed)
        self.rotationT = dim > 0 ? rotation.transposed() : rotation
        self.codebook = _codebook(dim, bits)
    }

    var descriptor: String { "mse:\(dim):\(bits):\(seed)" }

    func _quantizeUnitWithEstimate(_ unitVectors: MLXArray) -> (MLXArray, MLXArray) {
        if bits == 0 {
            return (
                MLXArray.zeros(Array(unitVectors.shape.dropLast()) + [0], dtype: .uint32),
                MLXArray.zeros(unitVectors.shape, dtype: .float32)
            )
        }
        let rotated = matmul(unitVectors, rotationT)
        let distances = abs(expandedDimensions(rotated, axis: -1) - codebook)
        let indices = argMin(distances, axis: -1).asType(.uint32)
        let packed = _packLowbit(indices, bits: bits)
        let estimatedRotated = take(codebook, indices.asType(.int32), axis: 0)
        return (packed, matmul(estimatedRotated, rotation))
    }

    func _quantizeUnit(_ unitVectors: MLXArray) -> MLXArray {
        _quantizeUnitWithEstimate(unitVectors).0
    }

    func _dequantizeUnit(_ packedIndices: MLXArray) -> MLXArray {
        if bits == 0 {
            return MLXArray.zeros(Array(packedIndices.shape.dropLast()) + [dim], dtype: .float32)
        }
        let indices = _unpackLowbit(packedIndices, bits: bits, length: dim).asType(.int32)
        let rotated = take(codebook, indices, axis: 0)
        return matmul(rotated, rotation)
    }

    func quantize(_ vectors: MLXArray) -> TurboQuantState {
        let vectorsF32 = vectors.asType(.float32)
        let norms = MLXLinalg.norm(vectorsF32, ord: 2, axis: -1)
        let safeNorms = maximum(expandedDimensions(norms, axis: -1), MLXArray(_turboQuantEps))
        let unitVectors = MLX.where(
            expandedDimensions(norms, axis: -1) .> 0,
            vectorsF32 / safeNorms,
            MLXArray.zeros(vectors.shape, dtype: .float32)
        )
        return .mse(TurboQuantMSEState(
            norms: norms.asType(vectors.dtype),
            indices: _quantizeUnit(unitVectors)
        ))
    }

    func dequantize(_ state: TurboQuantState) -> MLXArray {
        guard case .mse(let state) = state else {
            fatalError("TurboQuantMSECodec expected TurboQuantMSEState")
        }
        let unitVectors = _dequantizeUnit(state.indices)
        return expandedDimensions(state.norms.asType(.float32), axis: -1) * unitVectors
    }

    func prepareQueries(_ queries: MLXArray) -> TurboQuantPreparedQueries {
        .array(matmul(queries, rotationT))
    }

    func scorePrepared(_ preparedQueries: TurboQuantPreparedQueries, state: TurboQuantState)
        -> MLXArray
    {
        guard case .array(let preparedQueries) = preparedQueries,
            case .mse(let state) = state
        else {
            fatalError("TurboQuantMSECodec received incompatible prepared queries or state")
        }

        if preparedQueries.dim(-2) == 1 {
            let qRot = preparedQueries.reshaped(
                preparedQueries.dim(0), preparedQueries.dim(1), preparedQueries.dim(2),
                preparedQueries.dim(-1))
            if let fastScores = _metalMSEScore(qRot: qRot, state: state, bits: bits, codebook: codebook) {
                return fastScores
            }
        }

        let indices = _unpackLowbit(state.indices, bits: bits, length: dim).asType(.int32)
        let rotated = take(codebook, indices, axis: 0)
        let dots = einsum("bhmld,bhtd->bhmlt", preparedQueries, rotated)
        return dots * expandedDimensions(
            expandedDimensions(state.norms.asType(.float32), axis: 2), axis: 2)
    }

    func weightedSum(_ weights: MLXArray, state: TurboQuantState) -> MLXArray {
        guard case .mse(let state) = state else {
            fatalError("TurboQuantMSECodec expected TurboQuantMSEState")
        }
        if weights.dim(-2) == 1,
            let fastOutput = _metalMSEWeightedSum(
                weights: weights, state: state, bits: bits, codebook: codebook, rotation: rotation)
        {
            return fastOutput
        }

        let indices = _unpackLowbit(state.indices, bits: bits, length: dim).asType(.int32)
        let rotated = take(codebook, indices, axis: 0)
        let weightedRot = einsum(
            "bhmlt,bht,bhtd->bhmld",
            weights,
            state.norms.asType(.float32),
            rotated
        )
        return matmul(weightedRot, rotation)
    }

    func weightedSumFromScores(_ scores: MLXArray, state: TurboQuantState) -> MLXArray {
        guard case .mse(let mseState) = state else {
            fatalError("TurboQuantMSECodec expected TurboQuantMSEState")
        }
        if let fastOutput = _metalMSEWeightedSumFromScores(
            scores: scores, state: mseState, bits: bits, codebook: codebook, rotation: rotation)
        {
            return fastOutput
        }
        return weightedSum(softmax(scores, axis: -1), state: state)
    }

    func weightedSumStatsFromScores(_ scores: MLXArray, state: TurboQuantState)
        -> (MLXArray, MLXArray, MLXArray)
    {
        guard case .mse(let mseState) = state else {
            fatalError("TurboQuantMSECodec expected TurboQuantMSEState")
        }
        let maxScores = MLX.max(scores, axis: -1)
        if let fastOutput = _metalMSEWeightedSumSumFromScores(
            scores: scores, state: mseState, bits: bits, codebook: codebook, rotation: rotation)
        {
            let denom = sum(exp(scores - expandedDimensions(maxScores, axis: -1)), axis: -1)
            return (fastOutput, denom, maxScores)
        }

        let weights = exp(scores - expandedDimensions(maxScores, axis: -1))
        let output = weightedSum(weights, state: state)
        let denom = sum(weights, axis: -1)
        return (output, denom, maxScores)
    }
}

final class _TurboQuantProdCodec: _TurboQuantCodec {
    let dim: Int
    let bits: Int
    let seed: Int
    let mseCodec: _TurboQuantMSECodec
    let projection: MLXArray
    let projectionT: MLXArray
    let queryTransformT: MLXArray
    let scale: Float
    let scaleArray: MLXArray

    init(_ dim: Int, _ bits: Int, seed: Int) {
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.mseCodec = _TurboQuantMSECodec(dim, max(bits - 1, 0), seed: seed)
        self.projection = _projectionMatrix(dim: dim, seed: seed + 1)
        self.projectionT = dim > 0 ? projection.transposed() : projection
        self.queryTransformT =
            if dim > 0 {
                concatenated([mseCodec.rotationT, projectionT], axis: -1)
            } else {
                MLXArray.zeros([0, 0], dtype: .float32)
            }
        self.scale = dim > 0 ? Float(sqrt(Double.pi / 2) / Double(dim)) : 0
        self.scaleArray = MLXArray([self.scale])
    }

    var descriptor: String { "prod:\(dim):\(bits):\(seed)" }

    func quantize(_ vectors: MLXArray) -> TurboQuantState {
        let vectorsF32 = vectors.asType(.float32)
        let norms = MLXLinalg.norm(vectorsF32, ord: 2, axis: -1)
        let safeNorms = maximum(expandedDimensions(norms, axis: -1), MLXArray(_turboQuantEps))
        let unitVectors = MLX.where(
            expandedDimensions(norms, axis: -1) .> 0,
            vectorsF32 / safeNorms,
            MLXArray.zeros(vectors.shape, dtype: .float32)
        )

        let (mseIndices, mseUnit) = mseCodec._quantizeUnitWithEstimate(unitVectors)
        let residual = unitVectors - mseUnit
        let residualNorms = MLXLinalg.norm(residual, ord: 2, axis: -1)
        let projected = matmul(residual, projectionT)
        let signs = MLX.where(projected .>= 0, MLXArray(1), MLXArray(0)).asType(.uint32)

        return .prod(TurboQuantProdState(
            norms: norms.asType(vectors.dtype),
            mseIndices: mseIndices,
            residualNorms: residualNorms.asType(vectors.dtype),
            qjlSigns: _packLowbit(signs, bits: 1)
        ))
    }

    func dequantize(_ state: TurboQuantState) -> MLXArray {
        guard case .prod(let state) = state else {
            fatalError("TurboQuantProdCodec expected TurboQuantProdState")
        }
        let mseUnit = mseCodec._dequantizeUnit(state.mseIndices)
        let signBits = _unpackLowbit(state.qjlSigns, bits: 1, length: dim).asType(.float32)
        let signs = signBits * 2.0 - 1.0
        let qjlUnit = scale
            * expandedDimensions(state.residualNorms.asType(.float32), axis: -1)
            * matmul(signs, projection)
        return expandedDimensions(state.norms.asType(.float32), axis: -1) * (mseUnit + qjlUnit)
    }

    func prepareQueries(_ queries: MLXArray) -> TurboQuantPreparedQueries {
        let transformed = matmul(queries, queryTransformT)
        return .pair(
            transformed[.ellipsis, ..<dim],
            transformed[.ellipsis, dim...]
        )
    }

    func scorePrepared(_ preparedQueries: TurboQuantPreparedQueries, state: TurboQuantState)
        -> MLXArray
    {
        guard case .pair(let mseQueries, let projQueries) = preparedQueries,
            case .prod(let state) = state
        else {
            fatalError("TurboQuantProdCodec received incompatible prepared queries or state")
        }

        if projQueries.dim(-2) == 1 {
            let qRot = mseQueries.reshaped(
                mseQueries.dim(0), mseQueries.dim(1), mseQueries.dim(2), mseQueries.dim(-1))
            let qProj = projQueries.reshaped(
                projQueries.dim(0), projQueries.dim(1), projQueries.dim(2), projQueries.dim(-1))
            if let fastScores = _metalProdScore(
                qRot: qRot,
                qProj: qProj,
                state: state,
                mseBits: mseCodec.bits,
                codebook: mseCodec.codebook,
                scale: scaleArray)
            {
                return fastScores
            }
        }

        let mseScore: MLXArray =
            if mseCodec.bits > 0 {
                mseCodec.scorePrepared(
                    .array(mseQueries),
                    state: .mse(TurboQuantMSEState(norms: state.norms, indices: state.mseIndices)))
            } else {
                MLXArray.zeros(
                    [projQueries.dim(0), projQueries.dim(1), projQueries.dim(2), projQueries.dim(3), state.norms.dim(2)],
                    dtype: .float32)
            }

        if projQueries.dim(-2) == 1,
            let fastQJL = _metalQJLScore(qProj: projQueries.reshaped(
                projQueries.dim(0), projQueries.dim(1), projQueries.dim(2), projQueries.dim(-1)),
                state: state,
                scale: scaleArray)
        {
            return mseScore + fastQJL
        }

        let signBits = _unpackLowbit(state.qjlSigns, bits: 1, length: dim).asType(.float32)
        let signs = signBits * 2.0 - 1.0
        let qjlScore = scale
            * expandedDimensions(expandedDimensions(state.residualNorms.asType(.float32), axis: 2), axis: 2)
            * einsum("bhmld,bhtd->bhmlt", projQueries, signs)
        let norms = expandedDimensions(expandedDimensions(state.norms.asType(.float32), axis: 2), axis: 2)
        return mseScore + norms * qjlScore
    }
}

private func _selectOutlierIndices(_ tensor: MLXArray, avgBits: Double) -> (MLXArray, MLXArray) {
    let lowerBits = Int(floor(avgBits))
    let upperBits = Int(ceil(avgBits))
    precondition(lowerBits != upperBits, "Mixed-precision selection requires a fractional bit-width.")

    let dim = tensor.dim(-1)
    var highCount = Int(round((avgBits - Double(lowerBits)) * Double(dim) / Double(upperBits - lowerBits)))
    highCount = max(1, min(dim - 1, highCount))

    let scores = mean(abs(tensor.asType(.float32)), axes: [0, 1, 2])
    let order = argSort(scores, axis: 0).asArray(Int32.self)
    let lowValues = Array(order.dropLast(highCount)).sorted()
    let highValues = Array(order.suffix(highCount)).sorted()
    return (MLXArray(lowValues), MLXArray(highValues))
}

final class _SplitCodec: _TurboQuantCodec {
    let bits: Double
    let mode: _TurboQuantCodecMode
    let dim: Int
    let lowerBits: Int
    let upperBits: Int
    let seed: Int
    let lowIdx: MLXArray
    let highIdx: MLXArray
    let restoreOrder: MLXArray
    let lowCodec: _TurboQuantCodec
    let highCodec: _TurboQuantCodec

    init(tensor: MLXArray, bits: Double, mode: _TurboQuantCodecMode, seed: Int) {
        self.bits = bits
        self.mode = mode
        self.dim = tensor.dim(-1)
        self.lowerBits = Int(floor(bits))
        self.upperBits = Int(ceil(bits))
        self.seed = seed
        let (lowIdx, highIdx) = _selectOutlierIndices(tensor, avgBits: bits)
        self.lowIdx = lowIdx.asType(.int32)
        self.highIdx = highIdx.asType(.int32)
        let concatOrder = concatenated([self.lowIdx, self.highIdx], axis: 0)
        self.restoreOrder = argSort(concatOrder, axis: 0)

        switch mode {
        case .prod:
            self.lowCodec = _TurboQuantProdCodec(self.lowIdx.dim(0), lowerBits, seed: seed)
            self.highCodec = _TurboQuantProdCodec(self.highIdx.dim(0), upperBits, seed: seed + 97)
        case .mse:
            self.lowCodec = _TurboQuantMSECodec(self.lowIdx.dim(0), lowerBits, seed: seed)
            self.highCodec = _TurboQuantMSECodec(self.highIdx.dim(0), upperBits, seed: seed + 97)
        }
    }

    init(lowIdx: MLXArray, highIdx: MLXArray, bits: Double, mode: _TurboQuantCodecMode, seed: Int) {
        self.bits = bits
        self.mode = mode
        self.dim = lowIdx.dim(0) + highIdx.dim(0)
        self.lowerBits = Int(floor(bits))
        self.upperBits = Int(ceil(bits))
        self.seed = seed
        self.lowIdx = lowIdx.asType(.int32)
        self.highIdx = highIdx.asType(.int32)
        let concatOrder = concatenated([self.lowIdx, self.highIdx], axis: 0)
        self.restoreOrder = argSort(concatOrder, axis: 0)

        switch mode {
        case .prod:
            self.lowCodec = _TurboQuantProdCodec(self.lowIdx.dim(0), lowerBits, seed: seed)
            self.highCodec = _TurboQuantProdCodec(self.highIdx.dim(0), upperBits, seed: seed + 97)
        case .mse:
            self.lowCodec = _TurboQuantMSECodec(self.lowIdx.dim(0), lowerBits, seed: seed)
            self.highCodec = _TurboQuantMSECodec(self.highIdx.dim(0), upperBits, seed: seed + 97)
        }
    }

    var descriptor: String { "split:\(mode.rawValue):\(dim):\(bits):\(seed)" }
    var serializationArrays: [MLXArray] { [lowIdx, highIdx] }

    func quantize(_ vectors: MLXArray) -> TurboQuantState {
        let lowTensor = take(vectors, lowIdx, axis: -1)
        let highTensor = take(vectors, highIdx, axis: -1)
        return .split(TurboQuantSplitState(
            low: lowCodec.quantize(lowTensor),
            high: highCodec.quantize(highTensor)
        ))
    }

    func dequantize(_ state: TurboQuantState) -> MLXArray {
        guard case .split(let state) = state else {
            fatalError("SplitCodec expected TurboQuantSplitState")
        }
        let lowTensor = lowCodec.dequantize(state.low)
        let highTensor = highCodec.dequantize(state.high)
        let merged = concatenated([lowTensor, highTensor], axis: -1)
        return take(merged, restoreOrder.asType(.int32), axis: -1)
    }

    func prepareQueries(_ queries: MLXArray) -> TurboQuantPreparedQueries {
        let lowTensor = take(queries, lowIdx, axis: -1)
        let highTensor = take(queries, highIdx, axis: -1)
        return .split(lowCodec.prepareQueries(lowTensor), highCodec.prepareQueries(highTensor))
    }

    func scorePrepared(_ preparedQueries: TurboQuantPreparedQueries, state: TurboQuantState)
        -> MLXArray
    {
        guard case .split(let lowPrepared, let highPrepared) = preparedQueries,
            case .split(let state) = state
        else {
            fatalError("SplitCodec received incompatible prepared queries or state")
        }
        return lowCodec.scorePrepared(lowPrepared, state: state.low)
            + highCodec.scorePrepared(highPrepared, state: state.high)
    }

    func weightedSum(_ weights: MLXArray, state: TurboQuantState) -> MLXArray {
        guard case .split(let state) = state else {
            fatalError("SplitCodec expected TurboQuantSplitState")
        }
        let lowTensor = lowCodec.weightedSum(weights, state: state.low)
        let highTensor = highCodec.weightedSum(weights, state: state.high)
        let merged = concatenated([lowTensor, highTensor], axis: -1)
        return take(merged, restoreOrder.asType(.int32), axis: -1)
    }

    func weightedSumFromScores(_ scores: MLXArray, state: TurboQuantState) -> MLXArray {
        guard case .split(let state) = state else {
            fatalError("SplitCodec expected TurboQuantSplitState")
        }
        let lowTensor = lowCodec.weightedSumFromScores(scores, state: state.low)
        let highTensor = highCodec.weightedSumFromScores(scores, state: state.high)
        let merged = concatenated([lowTensor, highTensor], axis: -1)
        return take(merged, restoreOrder.asType(.int32), axis: -1)
    }

    func weightedSumStatsFromScores(_ scores: MLXArray, state: TurboQuantState)
        -> (MLXArray, MLXArray, MLXArray)
    {
        guard case .split(let state) = state else {
            fatalError("SplitCodec expected TurboQuantSplitState")
        }
        let (lowTensor, denom, maxScores) = lowCodec.weightedSumStatsFromScores(scores, state: state.low)
        let (highTensor, _, _) = highCodec.weightedSumStatsFromScores(scores, state: state.high)
        let merged = concatenated([lowTensor, highTensor], axis: -1)
        return (take(merged, restoreOrder.asType(.int32), axis: -1), denom, maxScores)
    }
}

func _buildCodec(_ tensor: MLXArray, bits: Double, mode: _TurboQuantCodecMode, seed: Int)
    -> _TurboQuantCodec
{
    let bits = try! _validateTurboBits(bits)
    if abs(bits - bits.rounded()) <= 1e-6 {
        switch mode {
        case .prod:
            return _TurboQuantProdCodec(tensor.dim(-1), Int(bits.rounded()), seed: seed)
        case .mse:
            return _TurboQuantMSECodec(tensor.dim(-1), Int(bits.rounded()), seed: seed)
        }
    }
    return _SplitCodec(tensor: tensor, bits: bits, mode: mode, seed: seed)
}

private func _parseCodecDescriptor(_ descriptor: String) -> [String] {
    descriptor.split(separator: ":").map(String.init)
}

private func _deserializeCodec(
    descriptor: String,
    arrays: ArraySlice<MLXArray>
) -> (_TurboQuantCodec, Int) {
    let parts = _parseCodecDescriptor(descriptor)
    guard let kind = parts.first else {
        fatalError("Invalid TurboQuant codec descriptor: \(descriptor)")
    }

    switch kind {
    case "mse":
        return (_TurboQuantMSECodec(Int(parts[1])!, Int(parts[2])!, seed: Int(parts[3])!), 0)
    case "prod":
        return (_TurboQuantProdCodec(Int(parts[1])!, Int(parts[2])!, seed: Int(parts[3])!), 0)
    case "split":
        let mode = _TurboQuantCodecMode(rawValue: parts[1])!
        let bits = Double(parts[3])!
        let seed = Int(parts[4])!
        let lowIdx = arrays[arrays.startIndex]
        let highIdx = arrays[arrays.index(after: arrays.startIndex)]
        return (_SplitCodec(lowIdx: lowIdx, highIdx: highIdx, bits: bits, mode: mode, seed: seed), 2)
    default:
        fatalError("Unsupported TurboQuant codec descriptor: \(descriptor)")
    }
}

private func _deserializeState(
    descriptor: String,
    arrays: ArraySlice<MLXArray>
) -> (TurboQuantState, Int) {
    let parts = _parseCodecDescriptor(descriptor)
    guard let kind = parts.first else {
        fatalError("Invalid TurboQuant codec descriptor: \(descriptor)")
    }

    switch kind {
    case "mse":
        let a0 = arrays[arrays.startIndex]
        let a1 = arrays[arrays.index(after: arrays.startIndex)]
        return (.mse(TurboQuantMSEState(norms: a0, indices: a1)), 2)
    case "prod":
        let i0 = arrays.startIndex
        let i1 = arrays.index(after: i0)
        let i2 = arrays.index(after: i1)
        let i3 = arrays.index(after: i2)
        return (
            .prod(TurboQuantProdState(
                norms: arrays[i0],
                mseIndices: arrays[i1],
                residualNorms: arrays[i2],
                qjlSigns: arrays[i3]
            )),
            4
        )
    case "split":
        let mode = _TurboQuantCodecMode(rawValue: parts[1])!
        let childDescriptor =
            switch mode {
            case .mse: "mse:0:0:0"
            case .prod: "prod:0:0:0"
            }
        let (_, lowCount) = _deserializeState(descriptor: childDescriptor, arrays: arrays)
        let (lowState, _) = _deserializeState(descriptor: childDescriptor, arrays: arrays.prefix(lowCount))
        let (highState, highCount) = _deserializeState(
            descriptor: childDescriptor, arrays: arrays.dropFirst(lowCount))
        return (.split(TurboQuantSplitState(low: lowState, high: highState)), lowCount + highCount)
    default:
        fatalError("Unsupported TurboQuant state descriptor: \(descriptor)")
    }
}

public final class TurboKVCache: BaseKVCache, CustomDebugStringConvertible {
    public let bits: Double
    public let seed: Int

    let decodeKeyChunkSize = 65_536
    let prefillKeyChunkSize = 512
    let prefillQueryBlockSize = 16
    let cacheStep = 256

    private var keys: TurboQuantState?
    private var values: TurboQuantState?
    private var keyCodec: _TurboQuantCodec?
    private var valueCodec: _TurboQuantCodec?
    private var pendingStateArrays: [MLXArray]?
    private var keyCodecDescriptor: String = ""
    private var valueCodecDescriptor: String = ""

    public init(bits: Double, seed: Int = defaultTurboQuantSeed) {
        self.bits = try! _validateTurboBits(bits)
        self.seed = seed
        super.init()
    }

    public static func fromCache(
        _ cache: KVCache,
        bits: Double,
        seed: Int = defaultTurboQuantSeed
    ) -> TurboKVCache {
        let turboCache = TurboKVCache(bits: bits, seed: seed)
        if cache.state.count >= 2 {
            _ = turboCache.updateAndFetch(keys: cache.state[0], values: cache.state[1])
        }
        return turboCache
    }

    public var nbytes: Int {
        _stateNBytes(_sliceState(keys, end: offset)) + _stateNBytes(_sliceState(values, end: offset))
    }

    public func dequantizedState() -> (MLXArray, MLXArray) {
        dequantize()
    }

    public func attention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
    ) -> MLXArray {
        let (keysState, valuesState) = updateAndFetch(keys: keys, values: values)
        if queries.dim(2) == 1 {
            return decodeAttention(
                queries: queries,
                keysState: keysState,
                valuesState: valuesState,
                scale: scale,
                mask: mask
            )
        }
        return quantizedAttention(
            queries: queries,
            keysState: keysState,
            valuesState: valuesState,
            scale: scale,
            mask: mask
        )
    }

    private func ensureCodecs(keys: MLXArray, values: MLXArray) {
        if keyCodec == nil {
            keyCodec = _buildCodec(keys, bits: bits, mode: .prod, seed: seed)
            keyCodecDescriptor = keyCodec?.descriptor ?? ""
        }
        if valueCodec == nil {
            valueCodec = _buildCodec(values, bits: bits, mode: .mse, seed: seed + 1)
            valueCodecDescriptor = valueCodec?.descriptor ?? ""
        }
    }

    func updateAndFetch(keys: MLXArray, values: MLXArray) -> (TurboQuantState, TurboQuantState) {
        ensureCodecs(keys: keys, values: values)
        let newKeys = keyCodec!.quantize(keys)
        let newValues = valueCodec!.quantize(values)

        let newEnd = offset + keys.dim(2)
        if self.keys == nil {
            self.keys = _allocateStateLike(newKeys, length: newEnd)
            self.values = _allocateStateLike(newValues, length: newEnd)
        } else {
            self.keys = _reserveStateCapacity(self.keys, used: offset, needed: newEnd, step: cacheStep)
            self.values = _reserveStateCapacity(self.values, used: offset, needed: newEnd, step: cacheStep)
        }

        _writeState(&self.keys!, source: newKeys, start: offset)
        _writeState(&self.values!, source: newValues, start: offset)
        offset = newEnd
        return (self.currentKeysState!, self.currentValuesState!)
    }

    var currentKeysState: TurboQuantState? {
        _sliceState(keys, end: offset)
    }

    var currentValuesState: TurboQuantState? {
        _sliceState(values, end: offset)
    }

    func dequantize(
        _ keysState: TurboQuantState? = nil,
        _ valuesState: TurboQuantState? = nil
    ) -> (MLXArray, MLXArray) {
        let keysState = keysState ?? currentKeysState!
        let valuesState = valuesState ?? currentValuesState!
        return (
            keyCodec!.dequantize(keysState).asType(.float32),
            valueCodec!.dequantize(valuesState).asType(.float32)
        )
    }

    private func applyAttentionMask(
        _ scores: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        qStart: Int,
        qEnd: Int,
        kStart: Int,
        kEnd: Int,
        totalQueries: Int,
        totalTokens: Int
    ) -> MLXArray {
        switch mask {
        case .none:
            return scores
        case .causal:
            let pastTokens = totalTokens - totalQueries
            let qIdx = MLXArray(Int32(pastTokens + qStart) ..< Int32(pastTokens + qEnd))
            let kIdx = MLXArray(Int32(kStart) ..< Int32(kEnd))
            let causalMask = expandedDimensions(qIdx, axis: -1) .>= expandedDimensions(kIdx, axis: -2)
            let expandedMask = expandedDimensions(causalMask, axes: [0, 1, 2])
            return MLX.where(expandedMask, scores, MLXArray(-Float.greatestFiniteMagnitude))
        case .array(let maskArray):
            var maskChunk = maskArray[.ellipsis, qStart ..< qEnd, kStart ..< kEnd]
            if maskChunk.ndim == scores.ndim - 1 {
                maskChunk = expandedDimensions(maskChunk, axis: 2)
            }
            if maskChunk.dtype == .bool {
                return MLX.where(maskChunk, scores, MLXArray(-Float.greatestFiniteMagnitude))
            }
            return scores + maskChunk
        case .arrays(let maskArrays):
            guard let maskArray = maskArrays.first else { return scores }
            var maskChunk = maskArray[.ellipsis, qStart ..< qEnd, kStart ..< kEnd]
            if maskChunk.ndim == scores.ndim - 1 {
                maskChunk = expandedDimensions(maskChunk, axis: 2)
            }
            if maskChunk.dtype == .bool {
                return MLX.where(maskChunk, scores, MLXArray(-Float.greatestFiniteMagnitude))
            }
            return scores + maskChunk
        }
    }

    func quantizedAttention(
        queries: MLXArray,
        keysState: TurboQuantState? = nil,
        valuesState: TurboQuantState? = nil,
        scale: Float = 1.0,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
    ) -> MLXArray {
        let keysState = keysState ?? currentKeysState!
        let valuesState = valuesState ?? currentValuesState!

        let (b, nQHeads, l, d) = (queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3))
        let nKVHeads = _kvHeadCount(keysState)
        let nRepeats = nQHeads / nKVHeads

        let groupedQueries = (queries * scale).reshaped(b, nKVHeads, nRepeats, l, d)
        let valueDim = valueCodec!.dim
        let totalTokens = _stateLength(keysState)
        let keyChunkSize = l == 1 ? decodeKeyChunkSize : prefillKeyChunkSize
        let queryBlockSize = l == 1 ? 1 : prefillQueryBlockSize

        var outputs: [MLXArray] = []
        for qStart in stride(from: 0, to: l, by: queryBlockSize) {
            let qEnd = min(l, qStart + queryBlockSize)
            let qBlock = groupedQueries[.ellipsis, qStart ..< qEnd, 0...]
            let preparedQueries = keyCodec!.prepareQueries(qBlock)

            var output = MLXArray.zeros([b, nKVHeads, nRepeats, qEnd - qStart, valueDim], dtype: .float32)
            var normalizer = MLXArray.zeros([b, nKVHeads, nRepeats, qEnd - qStart], dtype: .float32)
            var maxScore = MLXArray.full(
                [b, nKVHeads, nRepeats, qEnd - qStart],
                values: MLXArray(-Float.infinity),
                dtype: .float32)

            for kStart in stride(from: 0, to: totalTokens, by: keyChunkSize) {
                let kEnd = min(totalTokens, kStart + keyChunkSize)
                let keyChunk = _sliceStateRange(keysState, start: kStart, end: kEnd)!
                let valueChunk = _sliceStateRange(valuesState, start: kStart, end: kEnd)!

                var scores = keyCodec!.scorePrepared(preparedQueries, state: keyChunk)
                scores = applyAttentionMask(
                    scores, mask: mask, qStart: qStart, qEnd: qEnd, kStart: kStart, kEnd: kEnd,
                    totalQueries: l, totalTokens: totalTokens)

                let (chunkOutput, chunkDenom, chunkMax) = valueCodec!.weightedSumStatsFromScores(scores, state: valueChunk)
                let newMax = maximum(maxScore, chunkMax)
                let prevScale = MLX.exp(maxScore - newMax)
                let chunkScale = MLX.exp(chunkMax - newMax)

                output =
                    output * expandedDimensions(prevScale, axis: -1)
                    + chunkOutput * expandedDimensions(chunkScale, axis: -1)
                normalizer = normalizer * prevScale + chunkDenom * chunkScale
                maxScore = newMax
            }

            outputs.append(output / maximum(expandedDimensions(normalizer, axis: -1), MLXArray(_turboQuantEps)))
        }

        return concatenated(outputs, axis: 3).reshaped(b, nQHeads, l, valueDim).asType(queries.dtype)
    }

    private func compiledIntegerDecodeAttention(
        groupedQueries: MLXArray,
        keysState: TurboQuantState,
        valuesState: TurboQuantState
    ) -> MLXArray? {
        guard
            _metalAvailable(),
            let keyCodec = keyCodec as? _TurboQuantProdCodec,
            let valueCodec = valueCodec as? _TurboQuantMSECodec,
            keyCodec.bits == valueCodec.bits,
            keyCodec.mseCodec.bits > 0,
            case .prod(let keysState) = keysState,
            case .mse(let valuesState) = valuesState
        else { return nil }

        let bits = valueCodec.bits
        let decode = _compiledIntegerDecodeKernel(bits: bits)
        return decode([
            groupedQueries,
            keysState.norms,
            keysState.mseIndices,
            keysState.residualNorms,
            keysState.qjlSigns,
            valuesState.norms,
            valuesState.indices,
            keyCodec.queryTransformT,
            keyCodec.mseCodec.codebook,
            keyCodec.scaleArray,
            valueCodec.codebook,
            valueCodec.rotation,
        ])[0]
    }

    func decodeAttention(
        queries: MLXArray,
        keysState: TurboQuantState? = nil,
        valuesState: TurboQuantState? = nil,
        scale: Float = 1.0,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
    ) -> MLXArray {
        let keysState = keysState ?? currentKeysState!
        let valuesState = valuesState ?? currentValuesState!
        precondition(queries.dim(-2) == 1, "TurboQuant decode attention expects a single query token.")

        let (b, nQHeads, l, d) = (queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3))
        let nKVHeads = _kvHeadCount(keysState)
        let nRepeats = nQHeads / nKVHeads

        let groupedQueries = (queries * scale).reshaped(b, nKVHeads, nRepeats, l, d)
        let valueDim = valueCodec!.dim
        let totalTokens = _stateLength(keysState)

        let fastPathMaskAllowed: Bool =
            switch mask {
            case .none, .causal:
                true
            default:
                false
            }

        if totalTokens <= decodeKeyChunkSize && fastPathMaskAllowed {
            if let fastOutput = compiledIntegerDecodeAttention(
                groupedQueries: groupedQueries, keysState: keysState, valuesState: valuesState)
            {
                return fastOutput.reshaped(b, nQHeads, l, valueDim).asType(queries.dtype)
            }

            let preparedQueries = keyCodec!.prepareQueries(groupedQueries)
            let scores = keyCodec!.scorePrepared(preparedQueries, state: keysState)
            let output = valueCodec!.weightedSumFromScores(scores, state: valuesState)
            return output.reshaped(b, nQHeads, l, valueDim).asType(queries.dtype)
        }

        return quantizedAttention(
            queries: queries, keysState: keysState, valuesState: valuesState, scale: scale, mask: mask)
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let (keysState, valuesState) = updateAndFetch(keys: keys, values: values)
        return dequantize(keysState, valuesState)
    }

    public override func innerState() -> [MLXArray] {
        state
    }

    public override var state: [MLXArray] {
        get {
            guard let keysState = currentKeysState, let valuesState = currentValuesState else {
                return []
            }
            guard let keyCodec, let valueCodec else { return [] }
            return keyCodec.serializationArrays + _flattenState(keysState)
                + valueCodec.serializationArrays + _flattenState(valuesState)
        }
        set {
            guard !newValue.isEmpty else {
                keys = nil
                values = nil
                keyCodec = nil
                valueCodec = nil
                keyCodecDescriptor = ""
                valueCodecDescriptor = ""
                pendingStateArrays = nil
                offset = 0
                return
            }
            guard !keyCodecDescriptor.isEmpty, !valueCodecDescriptor.isEmpty else {
                pendingStateArrays = newValue
                return
            }

            var index = 0
            let (keyCodec, keyCodecArrayCount) = _deserializeCodec(
                descriptor: keyCodecDescriptor,
                arrays: ArraySlice(newValue[index...]))
            index += keyCodecArrayCount
            let (keysState, keyStateArrayCount) = _deserializeState(
                descriptor: keyCodecDescriptor,
                arrays: ArraySlice(newValue[index...]))
            index += keyStateArrayCount

            let (valueCodec, valueCodecArrayCount) = _deserializeCodec(
                descriptor: valueCodecDescriptor,
                arrays: ArraySlice(newValue[index...]))
            index += valueCodecArrayCount
            let (valuesState, valueStateArrayCount) = _deserializeState(
                descriptor: valueCodecDescriptor,
                arrays: ArraySlice(newValue[index...]))
            index += valueStateArrayCount

            precondition(index == newValue.count, "Unexpected TurboKVCache state payload size.")
            self.keyCodec = keyCodec
            self.valueCodec = valueCodec
            self.keys = keysState
            self.values = valuesState
            self.offset = _stateLength(keysState)
            self.pendingStateArrays = nil
        }
    }

    public override var metaState: [String] {
        get {
            [
                String(offset),
                String(bits),
                String(seed),
                keyCodec?.descriptor ?? keyCodecDescriptor,
                valueCodec?.descriptor ?? valueCodecDescriptor,
            ]
        }
        set {
            guard newValue.count == 5 else {
                fatalError("TurboKVCache metaState must have exactly 5 values")
            }
            offset = Int(newValue[0]) ?? 0
            keyCodecDescriptor = newValue[3]
            valueCodecDescriptor = newValue[4]
            if let pendingStateArrays {
                state = pendingStateArrays
            }
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    public var debugDescription: String {
        "TurboKVCache(offset: \(offset), bits: \(bits), seed: \(seed))"
    }
}

public func maybeQuantizeKVCache(
    cache: inout [KVCache],
    kvBits: Double?,
    quantizedKVStart: Int = 0,
    quantizationScheme: String? = nil
) {
    guard
        let kvBits,
        !cache.isEmpty,
        turboQuantEnabled(bits: kvBits, scheme: quantizationScheme)
    else { return }

    for index in 0 ..< cache.count {
        if cache[index] is TurboKVCache {
            continue
        }
        if let simpleCache = cache[index] as? KVCacheSimple, simpleCache.offset > quantizedKVStart {
            cache[index] = TurboKVCache.fromCache(simpleCache, bits: kvBits)
        }
    }
}
