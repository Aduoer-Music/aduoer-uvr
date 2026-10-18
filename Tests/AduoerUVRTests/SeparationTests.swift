// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import CoreML
import XCTest
@testable import AduoerUVR

final class SeparationTests: XCTestCase {
    func testResampling48kMonoProduces44kStereo() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let input = directory.appendingPathComponent("mono.wav")
        try writeFixture(input, rate: 48000, channels: 1, frames: 48000)
        let file = try AudioNormalization.open(input, directory: directory)
        XCTAssertEqual(file.processingFormat.sampleRate, 44100)
        XCTAssertEqual(file.processingFormat.channelCount, 2)
        XCTAssertEqual(Double(file.length), 44100, accuracy: 2)
    }

    func testCancellationBeforeModelLoad() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await SourceSeparator().separate(input: directory.appendingPathComponent("missing.wav"),
                directory: directory, modelURL: directory.appendingPathComponent("missing.mlmodelc"), configuration: .instHQ3)
        }
        do { _ = try await task.value; XCTFail("Expected cancellation") }
        catch is CancellationError { XCTAssertFalse(FileManager.default.fileExists(atPath: directory.appendingPathComponent("vocals.wav").path)) }
    }

    func testConfigurationRejectsInsufficientWindow() throws {
        let config = ModelConfiguration(id: "too-short", fft: 6144, hop: 1024, bins: 2048,
                                        timeFrames: 16, primaryStem: .vocals, outputName: "vocals")
        XCTAssertThrowsError(try UVRSpectrogram(configuration: config))
    }

    func testBothFrequencyShapesPreserveLowBandRoundTrip() throws {
        let frames = 220500
        let pcm = fixture(rate: 44100, channels: 2, frames: frames)
        for config in [ModelConfiguration.mdx9482, .instHQ3] {
            let transform = try UVRSpectrogram(configuration: config)
            let spectrum = try transform.encode(pcm.floatChannelData!, frames: frames)
            let restored = try transform.decode(spectrum, frames: frames)
            XCTAssertEqual(spectrum.shape.map(\.intValue), [1, 4, config.bins, 256])
            // Boundary truncation causes local ringing; evaluate the interior.
            var error: Double = 0
            for channel in 0..<2 {
                for index in 6144..<(frames-6144) {
                    error += pow(Double(restored[channel][index] - pcm.floatChannelData![channel][index]), 2)
                }
            }
            XCTAssertLessThan(sqrt(error / Double(2*(frames-12288))), 1e-5)
        }
    }

    func testMDX9482StreamingRealModel() async throws { try await streaming(configuration: .mdx9482, name: "UVR9482FP16") }
    func testInstHQ3StreamingRealModel() async throws { try await streaming(configuration: .instHQ3, name: "UVRInstHQ3FP16") }

    /// Verify non-full tail, overlap, reconstruction AND the model's primary stem routing.
    private func streaming(configuration: ModelConfiguration, name: String) async throws {
        guard let location = ProcessInfo.processInfo.environment["UVR_TEST_MODEL_DIR"] else {
            throw XCTSkip("Set UVR_TEST_MODEL_DIR to downloaded/converted model packages for integration tests")
        }
        let package = URL(fileURLWithPath: location).appendingPathComponent(name+".mlpackage")
        let model = try await MLModel.compileModel(at: package)
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let frames = 358533
        let input = directory.appendingPathComponent("input.wav")
        try writeFixture(input, rate: 44100, channels: 2, frames: frames)
        let result = try await SourceSeparator().separate(input: input, directory: directory, modelURL: model, configuration: configuration)
        let original = try read(input), vocals = try read(result.vocals), instrumental = try read(result.instrumental)
        XCTAssertEqual(vocals.frameLength, AVAudioFrameCount(frames))
        XCTAssertEqual(instrumental.frameLength, AVAudioFrameCount(frames))
        for channel in 0..<2 {
            for index in 0..<frames {
                let a = vocals.floatChannelData![channel][index], b = instrumental.floatChannelData![channel][index]
                guard a.isFinite && b.isFinite && abs(a+b-original.floatChannelData![channel][index]) < 1e-6 else {
                    XCTFail("Nonfinite or unaligned output at \(index)"); return
                }
            }
        }
        let first = fixture(rate: 44100, channels: 2, frames: 198450)
        let direct = try NativeCoreMLSourceModel(url: model, configuration: configuration, computeUnits: .all).predict(first)
        let primary = configuration.primaryStem == .vocals ? vocals : instrumental
        var maxError: Float = 0
        for channel in 0..<2 {
            for index in 0..<154350 { maxError = max(maxError, abs(direct[channel][index]-primary.floatChannelData![channel][index])) }
        }
        XCTAssertLessThan(maxError, 1e-5, "Primary target must route to the correct stem")
    }

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func fixture(rate: Double, channels: AVAudioChannelCount, frames: Int) -> AVAudioPCMBuffer {
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: channels, interleaved: false)!
        let pcm = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames))!
        pcm.frameLength = AVAudioFrameCount(frames)
        for channel in 0..<Int(channels) {
            for index in 0..<frames { pcm.floatChannelData![channel][index] = Float(0.3*sin(Double(index)*(channel==0 ? 440 : 523.25)*2 * .pi/rate)) }
        }
        return pcm
    }

    private func writeFixture(_ url: URL, rate: Double, channels: AVAudioChannelCount, frames: Int) throws {
        let pcm = fixture(rate: rate, channels: channels, frames: frames)
        let writer = try AVAudioFile(forWriting: url, settings: pcm.format.settings)
        try writer.write(from: pcm)
    }

    private func read(_ url: URL) throws -> AVAudioPCMBuffer {
        let file = try AVAudioFile(forReading: url)
        let pcm = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
        try file.read(into: pcm); return pcm
    }
}
