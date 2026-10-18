// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import Accelerate
import CoreML
import Foundation

public struct SeparatedAudioFiles: Sendable {
    public let vocals: URL
    public let instrumental: URL
    public let duration: Double
    public let elapsedSeconds: Double
}

/// Serial, bounded-memory inference. Supply a compiled .mlmodelc and matching descriptor.
public actor SourceSeparator {
    private let core = 176_400
    private let overlap = 22_050
    private let context = 22_050

    public init() {}

    /// Reads at most 5s at once; writes two aligned Float32 WAV files incrementally.
    /// The caller owns the output directory and removes it after failure or cancellation.
    public func separate(input: URL, directory: URL, modelURL: URL,
                         configuration: ModelConfiguration, computeUnits: MLComputeUnits = .all,
                         progress: @Sendable (Double) -> Void = { _ in }) throws -> SeparatedAudioFiles {
        try Task.checkCancellation()
        let started = ProcessInfo.processInfo.systemUptime
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let source = try AudioNormalization.open(input, directory: directory)
        let vocals = directory.appendingPathComponent("vocals.wav")
        let instrumental = directory.appendingPathComponent("instrumental.wav")
        do {
            let model = try NativeCoreMLSourceModel(url: modelURL, configuration: configuration, computeUnits: computeUnits)
            try process(source, vocals: vocals, instrumental: instrumental, configuration: configuration, progress: progress) {
                try model.predict($0)
            }
        }
        return SeparatedAudioFiles(vocals: vocals, instrumental: instrumental,
                                   duration: Double(source.length) / 44_100,
                                   elapsedSeconds: ProcessInfo.processInfo.systemUptime - started)
    }

    /// 每块仅提交不会被下一块覆盖的前缀，最后一块写完整尾部；保证无丢帧、无重复帧。
    func process(
        _ source: AVAudioFile, vocals: URL, instrumental: URL,
        configuration: ModelConfiguration, progress: @Sendable (Double) -> Void,
        predict: (AVAudioPCMBuffer) throws -> [[Float]]
    ) throws {
        let format = AudioNormalization.format
        let vocalWriter = try AVAudioFile(forWriting: vocals, settings: format.settings)
        let instrumentalWriter = try AVAudioFile(forWriting: instrumental, settings: format.settings)
        let frames = Int(source.length)
        var tail = [[Float]](repeating: [], count: 2)
        let fadeIn = (1...overlap).map { Float($0) / Float(overlap + 1) }
        let fadeOut = Array(fadeIn.reversed())
        var combinedWeights = [Float](repeating: 0, count: overlap)
        vDSP_vadd(fadeIn, 1, fadeOut, 1, &combinedWeights, 1, vDSP_Length(overlap))
        var start = 0
        progress(0)
        while start < frames {
            try Task.checkCancellation()
            let end = min(start + core, frames)
            let left = max(0, start - context)
            let right = min(frames, end + context)
            try autoreleasepool {
                guard let input = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(right - left)) else {
                    throw SeparationError.invalidAudio
                }
                source.framePosition = AVAudioFramePosition(left)
                try source.read(into: input, frameCount: input.frameCapacity)
                guard input.frameLength == right - left else {
                    throw SeparationError.incompleteRead(expected: right - left, actual: Int(input.frameLength),
                                                        position: Int64(left), length: source.length)
                }
                guard let channels = input.floatChannelData else { throw SeparationError.invalidAudio }
                let primary = try predict(input)
                try Task.checkCancellation()
                guard primary.count == 2, primary.allSatisfy({ $0.count == Int(input.frameLength) }) else {
                    throw SeparationError.invalidOutput
                }
                let count = end - start
                let written = end == frames ? count : count - overlap
                guard let vocalBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(written)),
                      let instrumentalBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(written)),
                      let vocalData = vocalBuffer.floatChannelData,
                      let instrumentalData = instrumentalBuffer.floatChannelData else { throw SeparationError.invalidOutput }
                vocalBuffer.frameLength = AVAudioFrameCount(written)
                instrumentalBuffer.frameLength = AVAudioFrameCount(written)
                var nextTail = [[Float]](repeating: [], count: 2)
                for channel in 0..<2 {
                    try primary[channel].withUnsafeBufferPointer { samples in
                        let values = samples.baseAddress!.advanced(by: start - left)
                        guard UnsafeBufferPointer(start: values, count: count).allSatisfy(\.isFinite) else {
                            throw SeparationError.invalidOutput
                        }
                        let primaryData = configuration.primaryStem == .vocals ? vocalData[channel] : instrumentalData[channel]
                        let residualData = configuration.primaryStem == .vocals ? instrumentalData[channel] : vocalData[channel]
                        primaryData.update(from: values, count: written)
                        if start > 0 {
                            vDSP_vmul(values, 1, fadeIn, 1, primaryData, 1, vDSP_Length(overlap))
                            vDSP_vadd(primaryData, 1, tail[channel], 1, primaryData, 1, vDSP_Length(overlap))
                            vDSP_vdiv(combinedWeights, 1, primaryData, 1, primaryData, 1, vDSP_Length(overlap))
                        }
                        vDSP_vsub(primaryData, 1, channels[channel].advanced(by: start - left), 1,
                                  residualData, 1, vDSP_Length(written))
                        if end < frames {
                            nextTail[channel] = [Float](repeating: 0, count: overlap)
                            vDSP_vmul(values.advanced(by: written), 1, fadeOut, 1,
                                      &nextTail[channel], 1, vDSP_Length(overlap))
                        }
                    }
                }
                try vocalWriter.write(from: vocalBuffer)
                try instrumentalWriter.write(from: instrumentalBuffer)
                tail = nextTail
            }
            progress(Double(end) / Double(frames))
            if end == frames { break }
            start += core - overlap
        }
    }
}
