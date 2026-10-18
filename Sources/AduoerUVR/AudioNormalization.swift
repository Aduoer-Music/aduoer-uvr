// SPDX-License-Identifier: Apache-2.0
import AVFoundation

public enum SeparationError: Error {
    case missingModel, invalidAudio, conversion, modelCreation, invalidOutput, invalidConfiguration
}

enum AudioNormalization {
    static let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 44_100, channels: 2, interleaved: false)!

    /// 44.1kHz 双声道直接按帧读取；其它规格通过有界缓冲转为临时 CAF，避免整曲 PCM 驻留内存。
    static func open(_ input: URL, directory: URL) throws -> AVAudioFile {
        let source = try AVAudioFile(forReading: input, commonFormat: .pcmFormatFloat32, interleaved: false)
        guard source.length > 0 else { throw SeparationError.invalidAudio }
        if source.processingFormat.sampleRate == format.sampleRate,
           source.processingFormat.channelCount == format.channelCount {
            return source
        }
        let destination = directory.appendingPathComponent("normalized.caf")
        try convert(source, to: destination)
        return try AVAudioFile(forReading: destination, commonFormat: .pcmFormatFloat32, interleaved: false)
    }

    /// 转换回调只提供请求的帧数，传递读盘错误并在输入结束后排空重采样器；每轮响应取消。
    private static func convert(_ source: AVAudioFile, to destination: URL) throws {
        guard let converter = AVAudioConverter(from: source.processingFormat, to: format),
              let input = AVAudioPCMBuffer(pcmFormat: source.processingFormat, frameCapacity: 8192),
              let output = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 8192) else {
            throw SeparationError.conversion
        }
        converter.downmix = true
        let writer = try AVAudioFile(forWriting: destination, settings: format.settings)
        while true {
            try Task.checkCancellation()
            var conversionError: NSError?
            var readError: Error?
            let status = converter.convert(to: output, error: &conversionError) { count, inputStatus in
                // AVAudioFile 在部分格式读到 EOF 时会返回 false 且没有 NSError；提前结束输入。
                let remaining = source.length - source.framePosition
                guard remaining > 0 else {
                    inputStatus.pointee = .endOfStream
                    return nil
                }
                do {
                    let frames = min(count, input.frameCapacity, AVAudioFrameCount(min(remaining, 8192)))
                    try source.read(into: input, frameCount: frames)
                    inputStatus.pointee = input.frameLength == 0 ? .endOfStream : .haveData
                    return input.frameLength == 0 ? nil : input
                } catch {
                    readError = error
                    inputStatus.pointee = .noDataNow
                    return nil
                }
            }
            if let readError { throw readError }
            if let conversionError { throw conversionError }
            guard status != .error else { throw SeparationError.conversion }
            if output.frameLength > 0 { try writer.write(from: output) }
            if status == .endOfStream { break }
        }
    }
}
