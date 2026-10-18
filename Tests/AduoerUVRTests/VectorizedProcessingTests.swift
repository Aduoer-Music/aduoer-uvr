// SPDX-License-Identifier: Apache-2.0
import Accelerate
import AVFoundation
import CoreML
import XCTest
@testable import AduoerUVR

final class VectorizedProcessingTests: XCTestCase {
    /// Check reflect padding, channel planes and frequency/time layout against a direct DFT.
    func testSpectrumMatchesDirectDFTAtBoundaries() throws {
        for config in [ModelConfiguration.mdx9482, .instHQ3] {
            for frames in [1, 6145, 220500] {
                let pcm = AVAudioPCMBuffer(pcmFormat: AudioNormalization.format, frameCapacity: AVAudioFrameCount(frames))!
                pcm.frameLength = AVAudioFrameCount(frames)
                for channel in 0..<2 {
                    for index in 0..<frames {
                        pcm.floatChannelData![channel][index] = Float(0.2 * cos(Double(index * (channel + 1)) * 0.137))
                    }
                }
                let spectrum = try UVRSpectrogram(configuration: config).encode(pcm.floatChannelData!, frames: frames)
                let values = spectrum.dataPointer.assumingMemoryBound(to: Float.self)
                let segment = config.hop * (config.timeFrames - 1)
                for channel in 0..<2 {
                    for time in [0, 1, 85, 255] {
                        for bin in [0, 1, 1023, config.bins - 1] {
                            var real = 0.0, imag = 0.0
                            for index in 0..<config.fft {
                                var position = time * config.hop + index - config.fft / 2
                                if position < 0 { position = -position }
                                if position >= segment { position = 2 * segment - 2 - position }
                                let sampleIndex = position - config.fft / 2
                                guard (0..<frames).contains(sampleIndex) else { continue }
                                let window = Float(0.5 - 0.5 * cos(2 * .pi * Double(index) / Double(config.fft)))
                                let sample = Double(pcm.floatChannelData![channel][sampleIndex] * window)
                                let phase = -2 * Double.pi * Double(bin * index) / Double(config.fft)
                                real += sample * cos(phase); imag += sample * sin(phase)
                            }
                            let offset = (channel * 2 * config.bins + bin) * config.timeFrames + time
                            XCTAssertEqual(Double(values[offset]), real, accuracy: 1e-3)
                            XCTAssertEqual(Double(values[offset + config.bins * config.timeFrames]), imag, accuracy: 1e-3)
                        }
                    }
                }
            }
        }
    }

    /// Core ML may return a non-contiguous tensor; decoding must honor every reported stride.
    func testDecodeAcceptsStridedSpectrum() throws {
        let frames = 10000
        let pcm = AVAudioPCMBuffer(pcmFormat: AudioNormalization.format, frameCapacity: AVAudioFrameCount(frames))!
        pcm.frameLength = AVAudioFrameCount(frames)
        for channel in 0..<2 {
            for index in 0..<frames { pcm.floatChannelData![channel][index] = Float(sin(Double(index) * 0.04)) }
        }
        for config in [ModelConfiguration.mdx9482, .instHQ3] {
            let transform = try UVRSpectrogram(configuration: config)
            let spectrum = try transform.encode(pcm.floatChannelData!, frames: frames)
            let expected = try transform.decode(spectrum, frames: frames)
            var storage = [Float](repeating: 0, count: spectrum.count * 2)
            try storage.withUnsafeMutableBufferPointer { buffer in
                vDSP_mmov(spectrum.dataPointer.assumingMemoryBound(to: Float.self), buffer.baseAddress!,
                          1, vDSP_Length(spectrum.count), 1, 2)
                let strided = try MLMultiArray(dataPointer: buffer.baseAddress!, shape: spectrum.shape,
                                              dataType: .float32, strides: spectrum.strides.map { NSNumber(value: $0.intValue * 2) })
                let actual = try transform.decode(strided, frames: frames)
                XCTAssertEqual(actual, expected)
            }
        }
    }

    /// Independent scalar overlap reference catches fade direction, normalization and final-tail errors.
    func testStreamingBlendMatchesScalarReference() async throws {
        let core = 176400, overlap = 22050, frames = 358533
        var sums = [Float](repeating: 0, count: frames)
        var weights = sums
        var start = 0, block = 0
        while start < frames {
            let end = min(start + core, frames)
            block += 1
            for index in 0..<(end - start) {
                var weight: Float = 1
                if start > 0 && index < overlap { weight *= Float(index + 1) / Float(overlap + 1) }
                if end < frames && end - start - index <= overlap { weight *= Float(end - start - index) / Float(overlap + 1) }
                sums[start + index] += Float(block) * 0.1 * weight
                weights[start + index] += weight
            }
            if end == frames { break }
            start += core - overlap
        }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let input = directory.appendingPathComponent("input.wav")
        let pcm = AVAudioPCMBuffer(pcmFormat: AudioNormalization.format, frameCapacity: AVAudioFrameCount(frames))!
        pcm.frameLength = AVAudioFrameCount(frames)
        for channel in 0..<2 { pcm.floatChannelData![channel].initialize(repeating: 0.5, count: frames) }
        do { try AVAudioFile(forWriting: input, settings: pcm.format.settings).write(from: pcm) }
        for config in [ModelConfiguration.mdx9482, .instHQ3] {
            let vocals = directory.appendingPathComponent("vocals.wav")
            let instrumental = directory.appendingPathComponent("instrumental.wav")
            var call = 0
            try await SourceSeparator().process(AVAudioFile(forReading: input), vocals: vocals, instrumental: instrumental,
                                                configuration: config, progress: { _ in }) { input in
                call += 1
                return [[Float]](repeating: [Float](repeating: Float(call) * 0.1, count: Int(input.frameLength)), count: 2)
            }
            for (url, primary) in [(vocals, config.primaryStem == .vocals), (instrumental, config.primaryStem == .instrumental)] {
                let file = try AVAudioFile(forReading: url)
                XCTAssertEqual(file.length, AVAudioFramePosition(frames))
                let result = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(frames))!
                try file.read(into: result)
                for channel in 0..<2 {
                    for index in 0..<frames {
                        let expected = sums[index] / weights[index]
                        guard abs(result.floatChannelData![channel][index] - (primary ? expected : 0.5 - expected)) < 1e-6 else {
                            XCTFail("Incorrect blend at \(index), channel \(channel), model \(config.id)"); return
                        }
                    }
                }
            }
        }
    }
}
