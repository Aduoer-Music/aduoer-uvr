// SPDX-License-Identifier: Apache-2.0
import AVFoundation

public enum SeparationError: Error, LocalizedError {
    case missingModel, invalidAudio, conversion, modelCreation, invalidOutput, invalidConfiguration
    case incompleteRead(expected: Int, actual: Int, position: Int64, length: Int64)

    public var errorDescription: String? {
        switch self {
        case .missingModel: return "The compiled separation model is missing."
        case .invalidAudio: return "The input contains no usable audio frames."
        case .conversion: return "Audio format conversion failed."
        case .modelCreation: return "The separation model could not be loaded."
        case .invalidOutput: return "The separation model returned invalid audio."
        case .invalidConfiguration: return "The model configuration is invalid."
        case let .incompleteRead(expected, actual, position, length):
            return "Incomplete PCM read: expected \(expected) frames, got \(actual), at \(position) of \(length)."
        }
    }
}

enum AudioNormalization {
    static let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 44_100, channels: 2, interleaved: false)!

    /// 仅目标规格的 PCM 可直接读取；压缩音频先顺序解码到 CAF，以实际解码长度替代估算长度。
    /// 转换使用有界缓冲，避免整曲 PCM 驻留内存，也避免对压缩音频反复 seek。
    static func open(_ input: URL, directory: URL) throws -> AVAudioFile {
        let source = try AVAudioFile(forReading: input, commonFormat: .pcmFormatFloat32, interleaved: false)
        guard source.length > 0 else { throw SeparationError.invalidAudio }
        if source.fileFormat.streamDescription.pointee.mFormatID == kAudioFormatLinearPCM,
           source.processingFormat.sampleRate == format.sampleRate,
           source.processingFormat.channelCount == format.channelCount {
            return source
        }
        let destination = directory.appendingPathComponent("normalized.caf")
        try convert(source, to: destination)
        let normalized = try AVAudioFile(forReading: destination, commonFormat: .pcmFormatFloat32, interleaved: false)
        guard normalized.length > 0 else { throw SeparationError.invalidAudio }
        return normalized
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
        var reachedEnd = false
        while true {
            try Task.checkCancellation()
            var conversionError: NSError?
            var readError: Error?
            let status = converter.convert(to: output, error: &conversionError) { count, inputStatus in
                guard !reachedEnd else {
                    inputStatus.pointee = .endOfStream
                    return nil
                }
                do {
                    let frames = min(count, input.frameCapacity)
                    input.frameLength = 0
                    try source.read(into: input, frameCount: frames)
                    // A short read is EOF, even when the compressed header advertises more frames.
                    reachedEnd = input.frameLength < frames
                    inputStatus.pointee = input.frameLength == 0 ? .endOfStream : .haveData
                    return input.frameLength == 0 ? nil : input
                } catch {
                    // AVAudioFile can report EOF as an OSStatus error on a subsequent read.
                    let failure = error as NSError
                    if failure.domain == NSOSStatusErrorDomain && failure.code == Int(kAudioFileEndOfFileError) {
                        reachedEnd = true
                        inputStatus.pointee = .endOfStream
                        return nil
                    }
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
