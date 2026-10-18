// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import CoreML
import XCTest
@testable import AduoerUVR

final class OracleValidationTests: XCTestCase {
    /// Compare the actual Swift FFT/ISTFT and full native waveform with independent Torch/ONNX oracles.
    func testInstHQ3AgainstONNXOracle() async throws { try await validate(.instHQ3, package: "UVRInstHQ3FP16") }
    func testMDX9482AgainstONNXOracle() async throws { try await validate(.mdx9482, package: "UVR9482FP16") }

    private func validate(_ configuration: ModelConfiguration, package: String) async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let oracleRoot = environment["UVR_ORACLE_DIR"], let modelRoot = environment["UVR_TEST_MODEL_DIR"] else {
            throw XCTSkip("Generate conversion oracles and set UVR_ORACLE_DIR plus UVR_TEST_MODEL_DIR")
        }
        let oracle = URL(fileURLWithPath: oracleRoot).appendingPathComponent(configuration.id)
        let input = try floats(oracle.appendingPathComponent("input.f32"))
        let spectrum = try floats(oracle.appendingPathComponent("spectrum.f32"))
        let predictedSpectrum = try floats(oracle.appendingPathComponent("output-spectrum.f32"))
        let referencePrimary = try floats(oracle.appendingPathComponent("primary.f32"))
        let frames = input.count / 2
        let pcm = AVAudioPCMBuffer(pcmFormat: AudioNormalization.format, frameCapacity: AVAudioFrameCount(frames))!
        pcm.frameLength = AVAudioFrameCount(frames)
        input.withUnsafeBufferPointer { data in
            for channel in 0..<2 { pcm.floatChannelData![channel].update(from: data.baseAddress!.advanced(by: channel * frames), count: frames) }
        }
        let transform = try UVRSpectrogram(configuration: configuration)
        let encoded = try transform.encode(pcm.floatChannelData!, frames: frames)
        let encodedValues = Array(UnsafeBufferPointer(start: encoded.dataPointer.assumingMemoryBound(to: Float.self), count: encoded.count))
        let encodeError = metrics(encodedValues, spectrum)
        XCTAssertLessThan(encodeError["relative_rms"]!, 1e-5)
        let output = try MLMultiArray(shape: encoded.shape, dataType: .float32)
        predictedSpectrum.withUnsafeBufferPointer { output.dataPointer.assumingMemoryBound(to: Float.self).update(from: $0.baseAddress!, count: $0.count) }
        let decodeError = metrics(try transform.decode(output, frames: frames).flatMap { $0 }, referencePrimary)
        XCTAssertLessThan(decodeError["relative_rms"]!, 1e-5)
        let modelURL = try await MLModel.compileModel(at: URL(fileURLWithPath: modelRoot).appendingPathComponent(package+".mlpackage"))
        let model = try NativeCoreMLSourceModel(url: modelURL, configuration: configuration, computeUnits: .all)
        let actual = try model.predict(pcm).flatMap { $0 }
        XCTAssertTrue(actual.allSatisfy { $0.isFinite })
        let waveformError = metrics(actual, referencePrimary)
        XCTAssertLessThan(waveformError["relative_rms"]!, 0.02)

    }

    private func floats(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    private func metrics(_ actual: [Float], _ expected: [Float]) -> [String: Double] {
        precondition(actual.count == expected.count)
        var squared = 0.0, power = 0.0, peak = 0.0
        for (a, b) in zip(actual, expected) {
            let delta = Double(a)-Double(b)
            squared += delta*delta; power += Double(b)*Double(b); peak = max(peak, abs(delta))
        }
        return ["relative_rms": sqrt(squared/max(power, 1e-20)), "max_abs": peak]
    }
}
