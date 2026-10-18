// SPDX-License-Identifier: Apache-2.0
import Accelerate
import CoreML
import Foundation

/// 与 sherpa UVR 对齐：FFT6144、periodic Hann、hop1024、center reflect、前后额外 trim3072。
/// 固定 batch=1 的 256 帧频谱最多容纳 254975 个输入样本，覆盖当前最长 5s 的块。
final class UVRSpectrogram {
    private let fft: Int
    private let hop: Int
    private let bins: Int
    private let times: Int
    private let segment: Int
    private let forward: vDSP_DFT_Setup
    private let inverse: vDSP_DFT_Setup
    private let window: [Float]
    private let denominator: [Float]

    init(configuration: ModelConfiguration) throws {
        let fft = configuration.fft
        let hop = configuration.hop
        let bins = configuration.bins
        let times = configuration.timeFrames
        guard fft > 0, hop > 0, bins > 0, bins <= fft / 2, times > 1,
              hop * (times - 1) - fft > 220_500 else { throw SeparationError.invalidConfiguration }
        self.fft = fft; self.hop = hop; self.bins = bins; self.times = times
        segment = hop * (times - 1)
        guard let forward = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(fft), .FORWARD) else {
            throw SeparationError.conversion
        }
        guard let inverse = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(fft), .INVERSE) else {
            vDSP_DFT_DestroySetup(forward)
            throw SeparationError.conversion
        }
        self.forward = forward; self.inverse = inverse
        let window = (0..<fft).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(fft))) }
        self.window = window
        var weights = [Float](repeating: 0, count: segment + fft)
        var squaredWindow = [Float](repeating: 0, count: fft)
        vDSP_vsq(window, 1, &squaredWindow, 1, vDSP_Length(fft))
        weights.withUnsafeMutableBufferPointer { buffer in
            for time in 0..<times {
                let destination = buffer.baseAddress!.advanced(by: time * hop)
                vDSP_vadd(destination, 1, squaredWindow, 1, destination, 1, vDSP_Length(fft))
            }
        }
        denominator = weights
    }

    deinit {
        vDSP_DFT_DestroySetup(forward)
        vDSP_DFT_DestroySetup(inverse)
    }

    /// 输入始终按两声道处理；保持与既有实现相同的补零及频率裁剪，不修改输入幅度。
    func encode(_ channels: UnsafePointer<UnsafeMutablePointer<Float>>, frames: Int) throws -> MLMultiArray {
        guard frames > 0 && frames < segment - fft else { throw SeparationError.invalidAudio }
        let tensor = try MLMultiArray(shape: [1, 4, NSNumber(value: bins), NSNumber(value: times)], dataType: .float32)
        let destination = tensor.dataPointer.assumingMemoryBound(to: Float.self)
        var real = [Float](repeating: 0, count: fft)
        let zero = [Float](repeating: 0, count: fft)
        var transformedReal = zero
        var transformedImag = zero
        for channel in 0..<2 {
            // Materialize center-reflect padding once, then window and copy complete vectors.
            // Keeping the per-sample work in Accelerate also makes ordinary Debug builds fast.
            var padded = [Float](repeating: 0, count: segment + fft)
            padded.withUnsafeMutableBufferPointer { buffer in
                let base = buffer.baseAddress!
                let half = fft / 2
                base.advanced(by: fft).update(from: channels[channel], count: frames)
                base.update(from: base.advanced(by: half + 1), count: half)
                vDSP_vrvrs(base, 1, vDSP_Length(half))
                let right = base.advanced(by: half + segment)
                right.update(from: base.advanced(by: segment - 1), count: half)
                vDSP_vrvrs(right, 1, vDSP_Length(half))
                for time in 0..<times {
                    vDSP_vmul(base.advanced(by: time * hop), 1, window, 1, &real, 1, vDSP_Length(fft))
                    vDSP_DFT_Execute(forward, real, zero, &transformedReal, &transformedImag)
                    vDSP_mmov(transformedReal, destination.advanced(by: channel * 2 * bins * times + time),
                              1, vDSP_Length(bins), 1, vDSP_Length(times))
                    vDSP_mmov(transformedImag, destination.advanced(by: (channel * 2 + 1) * bins * times + time),
                              1, vDSP_Length(bins), 1, vDSP_Length(times))
                }
            }
        }
        return tensor
    }

    /// 频谱高频补零后做逆变换/加窗重叠归一化，最后去除 center 与外层 trim，返回准确输入帧数。
    func decode(_ tensor: MLMultiArray, frames: Int) throws -> [[Float]] {
        guard tensor.dataType == .float32, tensor.shape.map(\.intValue) == [1, 4, bins, times] else {
            throw SeparationError.invalidOutput
        }
        let values = tensor.dataPointer.assumingMemoryBound(to: Float.self)
        let strides = tensor.strides.map(\.intValue)
        var real = [Float](repeating: 0, count: fft)
        var imag = real
        var waveform = real
        var ignoredImag = real
        var channels: [[Float]] = []
        for channel in 0..<2 {
            var accumulated = [Float](repeating: 0, count: denominator.count)
            accumulated.withUnsafeMutableBufferPointer { accumulation in
                for time in 0..<times {
                    let offset = time * strides[3]
                    vDSP_mmov(values.advanced(by: channel * 2 * strides[1] + offset), &real,
                              1, vDSP_Length(bins), vDSP_Length(strides[2]), 1)
                    vDSP_mmov(values.advanced(by: (channel * 2 + 1) * strides[1] + offset), &imag,
                              1, vDSP_Length(bins), vDSP_Length(strides[2]), 1)
                    imag[0] = 0
                    real.withUnsafeMutableBufferPointer { buffer in
                        let mirror = buffer.baseAddress!.advanced(by: fft - bins + 1)
                        mirror.update(from: buffer.baseAddress!.advanced(by: 1), count: bins - 1)
                        vDSP_vrvrs(mirror, 1, vDSP_Length(bins - 1))
                    }
                    imag.withUnsafeMutableBufferPointer { buffer in
                        let mirror = buffer.baseAddress!.advanced(by: fft - bins + 1)
                        vDSP_vneg(buffer.baseAddress!.advanced(by: 1), 1, mirror, 1, vDSP_Length(bins - 1))
                        vDSP_vrvrs(mirror, 1, vDSP_Length(bins - 1))
                    }
                    vDSP_DFT_Execute(inverse, real, imag, &waveform, &ignoredImag)
                    vDSP_vmul(waveform, 1, window, 1, &waveform, 1, vDSP_Length(fft))
                    var scale = Float(fft)
                    vDSP_vsdiv(waveform, 1, &scale, &waveform, 1, vDSP_Length(fft))
                    let destination = accumulation.baseAddress!.advanced(by: time * hop)
                    vDSP_vadd(destination, 1, waveform, 1, destination, 1, vDSP_Length(fft))
                }
            }
            var output = [Float](repeating: 0, count: frames)
            accumulated.withUnsafeBufferPointer { signal in
                denominator.withUnsafeBufferPointer { weights in
                    vDSP_vdiv(weights.baseAddress!.advanced(by: fft), 1, signal.baseAddress!.advanced(by: fft),
                              1, &output, 1, vDSP_Length(frames))
                }
            }
            guard output.allSatisfy(\.isFinite) else { throw SeparationError.invalidOutput }
            channels.append(output)
        }
        return channels
    }
}
