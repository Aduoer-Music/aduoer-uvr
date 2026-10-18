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
        for time in 0..<times {
            for index in 0..<fft { weights[time * hop + index] += window[index] * window[index] }
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
            var padded = [Float](repeating: 0, count: segment)
            padded.withUnsafeMutableBufferPointer { buffer in
                buffer.baseAddress!.advanced(by: fft / 2).update(from: channels[channel], count: frames)
            }
            for time in 0..<times {
                for index in 0..<fft {
                    var position = time * hop + index - fft / 2
                    if position < 0 { position = -position }
                    if position >= segment { position = 2 * segment - 2 - position }
                    real[index] = padded[position] * window[index]
                }
                vDSP_DFT_Execute(forward, real, zero, &transformedReal, &transformedImag)
                for frequency in 0..<bins {
                    destination[(channel * 2 * bins + frequency) * times + time] = transformedReal[frequency]
                    destination[((channel * 2 + 1) * bins + frequency) * times + time] = transformedImag[frequency]
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
            for time in 0..<times {
                for frequency in 0..<bins {
                    let offset = frequency * strides[2] + time * strides[3]
                    real[frequency] = values[channel * 2 * strides[1] + offset]
                    imag[frequency] = frequency == 0 ? 0 : values[(channel * 2 + 1) * strides[1] + offset]
                    if frequency > 0 {
                        real[fft - frequency] = real[frequency]
                        imag[fft - frequency] = -imag[frequency]
                    }
                }
                vDSP_DFT_Execute(inverse, real, imag, &waveform, &ignoredImag)
                for index in 0..<fft {
                    accumulated[time * hop + index] += waveform[index] * window[index] / Float(fft)
                }
            }
            var output = [Float](repeating: 0, count: frames)
            for index in 0..<frames {
                let position = fft + index
                output[index] = accumulated[position] / denominator[position]
                guard output[index].isFinite else { throw SeparationError.invalidOutput }
            }
            channels.append(output)
        }
        return channels
    }
}
