// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Must match the graph's tensor shape and original model's primary target.
public struct ModelConfiguration: Codable, Equatable, Sendable {
    public enum Stem: String, Codable, Sendable { case vocals, instrumental }
    public let id: String
    public let fft: Int
    public let hop: Int
    public let bins: Int
    public let timeFrames: Int
    public let primaryStem: Stem
    public let inputName: String
    public let outputName: String

    public init(id: String, fft: Int, hop: Int, bins: Int, timeFrames: Int,
                primaryStem: Stem, inputName: String = "spectrogram", outputName: String) {
        self.id = id; self.fft = fft; self.hop = hop; self.bins = bins
        self.timeFrames = timeFrames; self.primaryStem = primaryStem
        self.inputName = inputName; self.outputName = outputName
    }

    public static let mdx9482 = ModelConfiguration(id: "mdx9482", fft: 6144, hop: 1024, bins: 2048,
                                                  timeFrames: 256, primaryStem: .vocals, outputName: "vocals")
    public static let instHQ3 = ModelConfiguration(id: "instHQ3", fft: 6144, hop: 1024, bins: 3072,
                                                  timeFrames: 256, primaryStem: .instrumental, outputName: "primary")
}
