// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import CoreML

/// 固定形状的原生 FP16 ML Program；前后处理保留 float32，交由 Core ML 选择 CPU/GPU/ANE。
final class NativeCoreMLSourceModel {
    private let model: MLModel
    private let transform: UVRSpectrogram
    private let configuration: ModelConfiguration

    init(url: URL, configuration: ModelConfiguration, computeUnits: MLComputeUnits) throws {
        self.configuration = configuration
        let settings = MLModelConfiguration()
        settings.computeUnits = computeUnits
        model = try MLModel(contentsOf: url, configuration: settings)
        transform = try UVRSpectrogram(configuration: configuration)
        guard let input = model.modelDescription.inputDescriptionsByName[configuration.inputName]?.multiArrayConstraint,
              input.shape.map(\.intValue) == [1, 4, configuration.bins, configuration.timeFrames],
              model.modelDescription.outputDescriptionsByName[configuration.outputName] != nil else {
            throw SeparationError.invalidConfiguration
        }
    }

    func predict(_ input: AVAudioPCMBuffer) throws -> [[Float]] {
        guard let channels = input.floatChannelData else { throw SeparationError.invalidAudio }
        let spectrum = try transform.encode(channels, frames: Int(input.frameLength))
        let features = try MLDictionaryFeatureProvider(dictionary: [configuration.inputName: MLFeatureValue(multiArray: spectrum)])
        let result = try model.prediction(from: features)
        guard let output = result.featureValue(for: configuration.outputName)?.multiArrayValue else { throw SeparationError.invalidOutput }
        try Task.checkCancellation()
        return try transform.decode(output, frames: Int(input.frameLength))
    }
}
