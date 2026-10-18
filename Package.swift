// swift-tools-version: 5.9
// SPDX-License-Identifier: Apache-2.0
import PackageDescription

let package = Package(
    name: "AduoerUVR",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "AduoerUVR", targets: ["AduoerUVR"]),
    ],
    targets: [
        .target(name: "AduoerUVR", linkerSettings: [.linkedFramework("Accelerate"), .linkedFramework("CoreML")]),
        .testTarget(name: "AduoerUVRTests", dependencies: ["AduoerUVR"], resources: [.copy("Fixtures")]),
    ]
)
