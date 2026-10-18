// SPDX-License-Identifier: Apache-2.0
import CoreML
import CryptoKit
import Darwin
import Foundation

/// Downloads pinned package files to disk, verifies them, then compiles and atomically installs.
/// Callers should use Application Support and exclude the directory from backup.
public actor RemoteModelStore {
    public typealias Compiler = @Sendable (URL) async throws -> URL
    private struct Flight {
        let id = UUID()
        let task: Task<URL, Error>
        let progress: ModelProgressRelay
    }
    private let directory: URL
    private let session: URLSession
    private let compiler: Compiler
    private var flights: [String: Flight] = [:]
    private var removal: Task<Void, Error>?
    private var removalGeneration: UInt64 = 0

    public init(directory: URL, session: URLSession? = nil,
                compiler: @escaping Compiler = { try await MLModel.compileModel(at: $0) }) {
        self.directory = directory
        if let session { self.session = session }
        else {
            let config = URLSessionConfiguration.ephemeral
            config.httpShouldSetCookies = false
            config.urlCache = nil
            config.timeoutIntervalForResource = 600
            self.session = URLSession(configuration: config)
        }
        self.compiler = compiler
    }

    /// Concurrent requests share one installation. Cancellation cancels that shared preparation.
    /// Force downloading bypasses a ready installation, preserving it until its replacement is verified and compiled.
    public func prepare(_ model: RemoteModelDescriptor, baseURL: URL,
                        forceDownload: Bool = false,
                        progress: @escaping @Sendable (ModelPreparationProgress) -> Void = { _ in }) async throws -> URL {
        try Task.checkCancellation()
        guard removal == nil else { throw CancellationError() }
        let generation = removalGeneration
        try Self.validate(model, baseURL: baseURL)
        if !forceDownload, let cached = try cachedModelURL(for: model) {
            try Self.removePreviousVersions(of: model, keeping: cached.deletingLastPathComponent(), directory: directory)
            progress(.ready)
            return cached
        }
        let key = try Self.fingerprint(model)
        let flight: Flight
        if let current = flights[key] {
            current.progress.add(progress)
            flight = current
        } else {
            let relay = ModelProgressRelay(progress)
            let directory = directory, session = session, compiler = compiler
            let task = Task.detached(priority: .userInitiated) {
                try await Self.install(model, baseURL: baseURL, directory: directory,
                                       session: session, compiler: compiler, progress: relay)
            }
            flight = Flight(task: task, progress: relay)
            flights[key] = flight
        }
        do {
            let url = try await withTaskCancellationHandler {
                try await flight.task.value
            } onCancel: {
                flight.task.cancel()
            }
            guard generation == removalGeneration else { throw CancellationError() }
            if flights[key]?.id == flight.id { flights[key] = nil }
            try Task.checkCancellation()
            return url
        } catch {
            if flights[key]?.id == flight.id { flights[key] = nil }
            if Task.isCancelled || (error as? URLError)?.code == .cancelled { throw CancellationError() }
            throw error
        }
    }

    public func cancel(_ model: RemoteModelDescriptor) {
        guard let key = try? Self.fingerprint(model) else { return }
        flights[key]?.task.cancel()
    }

    /// Cancel and drain downloads/compilation before removing all installations and abandoned staging files.
    /// New preparations are rejected until removal completes, so a cancelled task cannot recreate deleted files.
    public func removeAll() async throws {
        if let removal { return try await removal.value }
        removalGeneration &+= 1
        let pending = Array(flights.values)
        pending.forEach { $0.task.cancel() }
        flights.removeAll()
        let directory = directory
        let task = Task.detached(priority: .utility) {
            for flight in pending { _ = await flight.task.result }
            if FileManager.default.fileExists(atPath: directory.path) {
                try FileManager.default.removeItem(at: directory)
            }
        }
        removal = task
        defer { removal = nil }
        try await task.value
    }

    /// Match completed installations by content, independent of release versions, asset names or download location.
    /// The receipt is written only after compilation; availability checks never move files, compile or download.
    public func cachedModelURL(for model: RemoteModelDescriptor) throws -> URL? {
        guard removal == nil else { return nil }
        let manager = FileManager.default
        guard manager.fileExists(atPath: directory.path) else { return nil }
        let key = try Self.fingerprint(model)
        for entry in try manager.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.isSymbolicLinkKey]) {
            let target = directory.appendingPathComponent(entry.lastPathComponent, isDirectory: true)
            guard !target.lastPathComponent.hasPrefix("."),
                  try target.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink != true,
                  let data = try? Data(contentsOf: target.appendingPathComponent("ready.json")),
                  let installed = try? JSONDecoder().decode(RemoteModelDescriptor.self, from: data),
                  try Self.fingerprint(installed) == key else { continue }
            let compiled = target.appendingPathComponent("model.mlmodelc", isDirectory: true)
            if manager.fileExists(atPath: compiled.appendingPathComponent("coremldata.bin").path) { return compiled }
        }
        return nil
    }

    private static func install(_ model: RemoteModelDescriptor, baseURL: URL, directory: URL,
                                session: URLSession, compiler: Compiler, progress: ModelProgressRelay) async throws -> URL {
        let manager = FileManager.default
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        var excluded = directory
        var values = URLResourceValues(); values.isExcludedFromBackup = true
        try excluded.setResourceValues(values)
        let staging = directory.appendingPathComponent(".preparing-\(UUID().uuidString)", isDirectory: true)
        try manager.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: staging) }
        let package = staging.appendingPathComponent(model.packageName, isDirectory: true)
        var finished: Int64 = 0
        for file in model.files {
            try Task.checkCancellation()
            let prior = finished
            progress.send(.downloading(Double(prior) / Double(model.downloadBytes)))
            let delegate = ModelFileDownloadDelegate(expectedBytes: file.bytes) { count in
                progress.send(.downloading(Double(prior + count) / Double(model.downloadBytes)))
            }
            var request = URLRequest(url: baseURL.appendingPathComponent(file.asset))
            request.timeoutInterval = 120
            let temporary: URL, response: URLResponse
            do {
                (temporary, response) = try await delegate.download(request, using: session)
            } catch {
                if delegate.exceededSize { throw ModelDownloadError.invalidSize(file.asset) }
                throw error
            }
            defer { try? manager.removeItem(at: temporary) }
            try Task.checkCancellation()
            guard let http = response as? HTTPURLResponse else { throw ModelDownloadError.httpStatus(0) }
            guard http.statusCode == 200 else { throw ModelDownloadError.httpStatus(http.statusCode) }
            let count = try temporary.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? -1
            guard Int64(count) == file.bytes else { throw ModelDownloadError.invalidSize(file.asset) }
            guard try sha256(temporary) == file.sha256 else { throw ModelDownloadError.checksumMismatch(file.asset) }
            let destination = package.appendingPathComponent(file.path)
            try manager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            try manager.moveItem(at: temporary, to: destination)
            finished += file.bytes
        }
        try Task.checkCancellation()
        progress.send(.compiling)
        let compiled = try await compiler(package)
        defer { try? manager.removeItem(at: compiled) }
        try Task.checkCancellation()
        guard manager.fileExists(atPath: compiled.appendingPathComponent("coremldata.bin").path) else {
            throw ModelDownloadError.invalidCompiledModel
        }
        try manager.moveItem(at: compiled, to: staging.appendingPathComponent("model.mlmodelc"))
        try manager.removeItem(at: package)
        try receipt(model).write(to: staging.appendingPathComponent("ready.json"), options: .atomic)
        let target = try target(model, directory: directory)
        try commitInstallation(from: staging, to: target)
        try removePreviousVersions(of: model, keeping: target, directory: directory)
        progress.send(.ready)
        return target.appendingPathComponent("model.mlmodelc", isDirectory: true)
    }

    /// Commit with one filesystem operation. A failed swap leaves the old installation intact;
    /// a successful swap leaves it at staging, where the installer's defer removes it.
    static func commitInstallation(from staging: URL, to target: URL) throws {
        let flags = FileManager.default.fileExists(atPath: target.path) ? RENAME_SWAP : RENAME_EXCL
        guard renamex_np(staging.path, target.path, UInt32(flags)) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
    }

    /// Only completed installations for the same model ID qualify; other models and active staging are untouched.
    private static func removePreviousVersions(of model: RemoteModelDescriptor, keeping target: URL,
                                              directory: URL) throws {
        let manager = FileManager.default
        for candidate in try manager.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.isSymbolicLinkKey]) {
            guard candidate.lastPathComponent != target.lastPathComponent,
                  !candidate.lastPathComponent.hasPrefix("."),
                  try candidate.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink != true,
                  let data = try? Data(contentsOf: candidate.appendingPathComponent("ready.json")),
                  let previous = try? JSONDecoder().decode(RemoteModelDescriptor.self, from: data),
                  previous.id == model.id else { continue }
            try manager.removeItem(at: candidate)
        }
    }

    private static func target(_ model: RemoteModelDescriptor, directory: URL) throws -> URL {
        directory.appendingPathComponent(try fingerprint(model), isDirectory: true)
    }

    private static func receipt(_ model: RemoteModelDescriptor) throws -> Data {
        let encoder = JSONEncoder(); encoder.outputFormatting = .sortedKeys
        return try encoder.encode(model)
    }

    private static func fingerprint(_ model: RemoteModelDescriptor) throws -> String {
        struct Content: Encodable {
            struct File: Encodable {
                let path: String
                let bytes: Int64
                let sha256: String
            }
            let id: String
            let files: [File]
        }
        let content = Content(id: model.id, files: model.files.sorted { $0.path < $1.path }.map {
            Content.File(path: $0.path, bytes: $0.bytes, sha256: $0.sha256)
        })
        let encoder = JSONEncoder(); encoder.outputFormatting = .sortedKeys
        return SHA256.hash(data: try encoder.encode(content)).map { String(format: "%02x", $0) }.joined()
    }

    private static func sha256(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hash = SHA256()
        while let data = try handle.read(upToCount: 262_144), !data.isEmpty {
            try Task.checkCancellation()
            hash.update(data: data)
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func validate(_ model: RemoteModelDescriptor, baseURL: URL) throws {
        func safe(_ path: String) -> Bool {
            !path.isEmpty && !path.hasPrefix("/") && !path.contains("\\") &&
            path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
        }
        guard baseURL.scheme == "https", baseURL.host != nil,
              baseURL.user == nil, baseURL.password == nil, baseURL.query == nil, baseURL.fragment == nil,
              safe(model.packageName), !model.packageName.contains("/"), model.packageName.hasSuffix(".mlpackage"),
              !model.files.isEmpty, model.files.count <= 16,
              Set(model.files.map(\.path)).count == model.files.count,
              model.files.contains(where: { $0.path == "Manifest.json" }),
              model.files.allSatisfy({ safe($0.path) && safe($0.asset) && !$0.asset.contains("/") &&
                  $0.bytes > 0 && $0.bytes <= 1_073_741_824 && $0.sha256.count == 64 &&
                  $0.sha256.allSatisfy { "0123456789abcdef".contains($0) } }) else {
            throw ModelDownloadError.invalidDescriptor
        }
    }
}

private final class ModelProgressRelay: @unchecked Sendable {
    private let lock = NSLock()
    private var callbacks: [@Sendable (ModelPreparationProgress) -> Void]
    private var latest = ModelPreparationProgress.downloading(0)
    init(_ callback: @escaping @Sendable (ModelPreparationProgress) -> Void) { callbacks = [callback] }
    func add(_ callback: @escaping @Sendable (ModelPreparationProgress) -> Void) {
        lock.lock(); callbacks.append(callback); let value = latest; lock.unlock()
        callback(value)
    }
    func send(_ value: ModelPreparationProgress) {
        lock.lock(); latest = value; let listeners = callbacks; lock.unlock()
        listeners.forEach { $0(value) }
    }
}

private final class ModelFileDownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private var oversized = false
    private var cancelled = false
    private var task: URLSessionDownloadTask?
    private var continuation: CheckedContinuation<(URL, URLResponse), Error>?
    private var downloadedFile: URL?
    private var fileError: Error?
    var exceededSize: Bool { lock.lock(); defer { lock.unlock() }; return oversized }
    let expectedBytes: Int64
    let progress: @Sendable (Int64) -> Void
    init(expectedBytes: Int64, progress: @escaping @Sendable (Int64) -> Void) {
        self.expectedBytes = expectedBytes; self.progress = progress
    }
    /// Use a delegate-backed task: the async convenience download API consumes download callbacks.
    /// Own the temporary file before Foundation removes it, and cancel the task with its caller.
    func download(_ request: URLRequest, using session: URLSession) async throws -> (URL, URLResponse) {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                lock.lock()
                guard !cancelled else {
                    lock.unlock()
                    continuation.resume(throwing: CancellationError())
                    return
                }
                let task = session.downloadTask(with: request)
                task.delegate = self
                self.task = task
                self.continuation = continuation
                lock.unlock()
                task.resume()
            }
        } onCancel: {
            self.lock.lock()
            self.cancelled = true
            let task = self.task
            self.lock.unlock()
            task?.cancel()
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        let destination = FileManager.default.temporaryDirectory.appendingPathComponent("uvr-model-\(UUID().uuidString)")
        do {
            try FileManager.default.moveItem(at: location, to: destination)
            lock.lock(); downloadedFile = destination; lock.unlock()
        } catch {
            lock.lock(); fileError = error; lock.unlock()
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        lock.lock()
        let continuation = self.continuation
        self.continuation = nil
        self.task = nil
        let file = downloadedFile
        downloadedFile = nil
        let failure = cancelled ? CancellationError() : (error ?? fileError)
        lock.unlock()
        if let failure {
            if let file { try? FileManager.default.removeItem(at: file) }
            continuation?.resume(throwing: failure)
        } else if let file, let response = task.response {
            continuation?.resume(returning: (file, response))
        } else {
            if let file { try? FileManager.default.removeItem(at: file) }
            continuation?.resume(throwing: URLError(.badServerResponse))
        }
    }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        if totalBytesWritten > expectedBytes {
            lock.lock(); oversized = true; lock.unlock()
            downloadTask.cancel(); return
        }
        progress(min(totalBytesWritten, expectedBytes))
    }
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(request.url?.scheme == "https" ? request : nil)
    }
}
