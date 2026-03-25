import Foundation
import MLX
@testable import MLXLMCommon
import Testing

private func seededNormal(_ shape: [Int], seed: UInt64) -> MLXArray {
    withRandomState(MLXRandom.RandomState(seed: seed)) {
        normal(shape, dtype: .float32)
    }
}

private func sampleUnitVectors(count: Int, dim: Int, seed: UInt64 = 0) -> MLXArray {
    let vectors = seededNormal([count, dim], seed: seed)
    let norms = MLXLinalg.norm(vectors, ord: 2, axis: -1, keepDims: true)
    return vectors / maximum(norms, MLXArray(1e-6 as Float))
}

private func approximatelyEqual(
    _ value: Float,
    _ target: Float,
    rel: Float,
    absTol: Float
) -> Bool {
    abs(value - target) <= max(absTol, rel * abs(target))
}

private func attentionScale(_ dim: Int) -> Float {
    Float(Foundation.pow(Double(dim), -0.5))
}

@Suite(.serialized)
struct TurboKVCacheTests {
    @Test
    func testTurboQuantMSEMatchesPaperSmallBitDistortions() async throws {
        let vectors = sampleUnitVectors(count: 256, dim: 64)
        let expected: [Int: Float] = [1: 0.36, 2: 0.117, 3: 0.03]

        for (bits, target) in expected {
            let codec = _TurboQuantMSECodec(64, bits, seed: 0)
            let state = codec.quantize(vectors)
            let reconstructed = codec.dequantize(state)
            let mse = mean(sum((vectors - reconstructed).square(), axis: -1)).item(Float.self)
            #expect(approximatelyEqual(mse, target, rel: 0.25, absTol: 0.02))
        }
    }

    @Test
    func testTurboQuantProdIsNearlyUnbiasedAcrossSeeds() async throws {
        let keys = sampleUnitVectors(count: 128, dim: 64, seed: 1)
        let queries = seededNormal([128, 64], seed: 2)
        let trueInnerProducts = sum(keys * queries, axis: -1)

        var estimates: [MLXArray] = []
        for seed in 0 ..< 16 {
            let codec = _TurboQuantProdCodec(64, 2, seed: seed)
            let state = codec.quantize(keys)
            let reconstructed = codec.dequantize(state)
            estimates.append(sum(reconstructed * queries, axis: -1))
        }

        let meanEstimate = mean(stacked(estimates, axis: 0), axis: 0)
        let bias = mean(meanEstimate - trueInnerProducts).item(Float.self)
        #expect(abs(bias) < 0.03)
    }

    @Test
    func testFractionalTurboQuantImprovesReconstruction() async throws {
        let vectors = seededNormal([1, 2, 32, 64], seed: 3)

        let codec3Bit = _buildCodec(vectors, bits: 3.0, mode: .mse, seed: 0)
        let codec35Bit = _buildCodec(vectors, bits: 3.5, mode: .mse, seed: 0)

        let state3Bit = codec3Bit.quantize(vectors)
        let state35Bit = codec35Bit.quantize(vectors)

        let mse3Bit = mean((vectors - codec3Bit.dequantize(state3Bit)).square()).item(Float.self)
        let mse35Bit = mean((vectors - codec35Bit.dequantize(state35Bit)).square()).item(Float.self)

        #expect(turboQuantEnabled(bits: 3.5))
        #expect(!turboQuantEnabled(bits: 3.0))
        #expect(mse35Bit < mse3Bit)
    }

    @Test
    func testTurboKVCacheConversionForFractionalBits() async throws {
        let layerCache = KVCacheSimple()
        _ = layerCache.update(
            keys: seededNormal([1, 2, 8, 32], seed: 4),
            values: seededNormal([1, 2, 8, 32], seed: 5)
        )
        var promptCache: [KVCache] = [layerCache]

        maybeQuantizeKVCache(
            cache: &promptCache,
            kvBits: 3.5,
            quantizedKVStart: 4
        )

        #expect(promptCache[0] is TurboKVCache)
    }

    @Test
    func testExplicitTurboQuantSchemeSupportsIntegerBits() async throws {
        let layerCache = KVCacheSimple()
        _ = layerCache.update(
            keys: seededNormal([1, 2, 8, 32], seed: 6),
            values: seededNormal([1, 2, 8, 32], seed: 7)
        )
        var promptCache: [KVCache] = [layerCache]

        maybeQuantizeKVCache(
            cache: &promptCache,
            kvBits: 3.0,
            quantizedKVStart: 4,
            quantizationScheme: "turboquant"
        )

        #expect(promptCache[0] is TurboKVCache)
    }

    @Test
    func testTurboQuantSkipsNonKVCacheEntries() async throws {
        let linearCache = ArraysCache(size: 2)
        linearCache[0] = MLXArray.zeros([1, 8], dtype: .float32)
        linearCache[1] = MLXArray.ones([1, 8], dtype: .float32)

        let attentionCache = KVCacheSimple()
        _ = attentionCache.update(
            keys: seededNormal([1, 2, 8, 32], seed: 8),
            values: seededNormal([1, 2, 8, 32], seed: 9)
        )

        var promptCache: [KVCache] = [linearCache, attentionCache]
        maybeQuantizeKVCache(
            cache: &promptCache,
            kvBits: 3.5,
            quantizedKVStart: 4,
            quantizationScheme: "turboquant"
        )

        #expect(promptCache[0] is ArraysCache)
        #expect(promptCache[1] is TurboKVCache)
    }

    @Test
    func testTurboKVCachePreservesAttentionShapeAndCompressesMemory() async throws {
        let keys = seededNormal([1, 2, 16, 32], seed: 10)
        let values = seededNormal([1, 2, 16, 32], seed: 11)
        let queries = seededNormal([1, 2, 1, 32], seed: 12)

        let fpCache = KVCacheSimple()
        let (fpKeys, fpValues) = fpCache.update(keys: keys, values: values)
        let reference = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: fpKeys,
            values: fpValues,
            scale: attentionScale(32),
            mask: .none
        )

        let turboCache = TurboKVCache.fromCache(fpCache, bits: 3.5)
        let quantized = turboCache.decodeAttention(
            queries: queries,
            scale: attentionScale(32),
            mask: .none
        )

        let diff = mean(abs(reference - quantized)).item(Float.self)
        #expect(quantized.shape == reference.shape)
        #expect(turboCache.nbytes < (fpKeys.nbytes + fpValues.nbytes))
        #expect(diff < 0.35)
    }

    @Test
    func testTurboKVDecodeAttentionMatchesDequantizedAttention() async throws {
        let keys = seededNormal([1, 2, 16, 32], seed: 13)
        let values = seededNormal([1, 2, 16, 32], seed: 14)
        let queries = seededNormal([1, 4, 1, 32], seed: 15)

        let fpCache = KVCacheSimple()
        _ = fpCache.update(keys: keys, values: values)
        let turboCache = TurboKVCache.fromCache(fpCache, bits: 3.5)
        let (dequantizedKeys, dequantizedValues) = turboCache.dequantizedState()

        let reference = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: dequantizedKeys.asType(queries.dtype),
            values: dequantizedValues.asType(queries.dtype),
            scale: attentionScale(32),
            mask: .none
        )
        let quantized = turboCache.decodeAttention(
            queries: queries,
            scale: attentionScale(32),
            mask: .none
        )

        let diff = MLX.max(abs(reference - quantized)).item(Float.self)
        #expect(quantized.shape == reference.shape)
        #expect(diff < 1e-4)
    }

    @Test
    func testTurboKVPrefillAttentionMatchesDequantizedAttention() async throws {
        let keys = seededNormal([1, 2, 12, 32], seed: 16)
        let values = seededNormal([1, 2, 12, 32], seed: 17)
        let queries = seededNormal([1, 4, 4, 32], seed: 18)

        let fpCache = KVCacheSimple()
        _ = fpCache.update(keys: keys, values: values)
        let turboCache = TurboKVCache.fromCache(fpCache, bits: 3.5)
        let (dequantizedKeys, dequantizedValues) = turboCache.dequantizedState()

        let reference = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: dequantizedKeys.asType(queries.dtype),
            values: dequantizedValues.asType(queries.dtype),
            scale: attentionScale(32),
            mask: .causal
        )
        let quantized = turboCache.quantizedAttention(
            queries: queries,
            scale: attentionScale(32),
            mask: .causal
        )

        let diff = MLX.max(abs(reference - quantized)).item(Float.self)
        #expect(quantized.shape == reference.shape)
        #expect(diff < 1e-4)
    }
}
