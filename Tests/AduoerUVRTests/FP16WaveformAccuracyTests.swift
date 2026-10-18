// SPDX-License-Identifier: Apache-2.0
import XCTest

/// Mixed relative/absolute error: a nearly silent target must not amplify FP16 rounding into a large ratio.
/// The absolute budget is relative to the input level, so quieter inputs still have a tighter bound.
struct FP16WaveformAccuracy: CustomStringConvertible {
    let errorRMS: Double
    let referenceRMS: Double
    let inputRMS: Double
    var limit: Double { 0.02 * referenceRMS + 0.0002 * inputRMS }
    var isAcceptable: Bool {
        errorRMS.isFinite && referenceRMS.isFinite && inputRMS.isFinite && errorRMS <= limit
    }
    var description: String {
        "error RMS \(errorRMS), limit \(limit), reference RMS \(referenceRMS), input RMS \(inputRMS)"
    }

    init(actual: [Float], reference: [Float], input: [Float]) {
        precondition(!actual.isEmpty && actual.count == reference.count && actual.count == input.count)
        var error = 0.0, referencePower = 0.0, inputPower = 0.0
        for index in actual.indices {
            let delta = Double(actual[index]) - Double(reference[index])
            error += delta * delta
            referencePower += Double(reference[index]) * Double(reference[index])
            inputPower += Double(input[index]) * Double(input[index])
        }
        let count = Double(actual.count)
        errorRMS = sqrt(error / count)
        referenceRMS = sqrt(referencePower / count)
        inputRMS = sqrt(inputPower / count)
    }
}

final class FP16WaveformAccuracyTests: XCTestCase {
    func testNearSilenceAllowsSmallInputRelativeRoundingButRejectsLostOutput() {
        let reference: [Float] = [0.0005, -0.0005]
        let input: [Float] = [0.2, -0.2]
        XCTAssertTrue(FP16WaveformAccuracy(actual: [0.00053, -0.00053], reference: reference, input: input).isAcceptable)
        XCTAssertFalse(FP16WaveformAccuracy(actual: [0, 0], reference: reference, input: input).isAcceptable)
        XCTAssertFalse(FP16WaveformAccuracy(actual: [0.001, -0.001], reference: reference, input: input).isAcceptable)
    }

    func testSignificantOutputStillRejectsThreePercentGainError() {
        XCTAssertFalse(FP16WaveformAccuracy(actual: [0.103, -0.103], reference: [0.1, -0.1], input: [0.2, -0.2]).isAcceptable)
        XCTAssertTrue(FP16WaveformAccuracy(actual: [0.101, -0.101], reference: [0.1, -0.1], input: [0.2, -0.2]).isAcceptable)
    }

    func testToleranceScalesWithInputAndRejectsNonfiniteValues() {
        XCTAssertFalse(FP16WaveformAccuracy(actual: [0.00003], reference: [0], input: [0.002]).isAcceptable)
        for invalid in [Float.nan, .infinity, -.infinity] {
            XCTAssertFalse(FP16WaveformAccuracy(actual: [invalid], reference: [0], input: [0.2]).isAcceptable)
            XCTAssertFalse(FP16WaveformAccuracy(actual: [0], reference: [invalid], input: [0.2]).isAcceptable)
            XCTAssertFalse(FP16WaveformAccuracy(actual: [0], reference: [0], input: [invalid]).isAcceptable)
        }
    }
}
