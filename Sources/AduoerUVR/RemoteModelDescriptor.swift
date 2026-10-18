// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Pin this catalog in the client. Do not trust an unverified network catalog to replace its hashes.
public struct ModelReleaseCatalog: Codable, Sendable {
    public let schemaVersion: Int
    public let release: String
    public let baseURL: URL
    public let models: [RemoteModelDescriptor]
}

public struct RemoteModelDescriptor: Codable, Sendable {
    public struct File: Codable, Sendable {
        public let path: String
        public let asset: String
        public let bytes: Int64
        public let sha256: String

        public init(path: String, asset: String, bytes: Int64, sha256: String) {
            self.path = path; self.asset = asset; self.bytes = bytes; self.sha256 = sha256
        }
    }

    public let id: String
    public let version: String
    public let packageName: String
    public let files: [File]
    public var downloadBytes: Int64 { files.reduce(0) { $0 + $1.bytes } }

    public init(id: String, version: String, packageName: String, files: [File]) {
        self.id = id; self.version = version; self.packageName = packageName; self.files = files
    }
}

public enum ModelPreparationProgress: Equatable, Sendable {
    case downloading(Double)
    case compiling
    case ready
}

public enum ModelDownloadError: Error, LocalizedError {
    case invalidDescriptor
    case httpStatus(Int)
    case invalidSize(String)
    case checksumMismatch(String)
    case invalidCompiledModel

    public var errorDescription: String? {
        switch self {
        case .invalidDescriptor: "Invalid model download descriptor"
        case .httpStatus(let status): "Model download failed (HTTP \(status))"
        case .invalidSize(let file): "Unexpected model file size: \(file)"
        case .checksumMismatch(let file): "Model checksum verification failed: \(file)"
        case .invalidCompiledModel: "Model compiler did not produce a compiled model"
        }
    }
}
