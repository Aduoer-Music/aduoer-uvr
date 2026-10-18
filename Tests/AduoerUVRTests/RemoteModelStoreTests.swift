// SPDX-License-Identifier: Apache-2.0
import CryptoKit
import Foundation
import XCTest
@testable import AduoerUVR

final class RemoteModelStoreTests: XCTestCase {
    private let baseURL = URL(string: "https://models.example/release/v1")!
    private let payload = Data("test-manifest".utf8)

    override func setUp() { super.setUp(); ModelAssetURLProtocol.reset() }

    func testVerifiedPackageCompilesAndReloadsOfflineWithoutRequests() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let session = session(); defer { session.invalidateAndCancel() }
        let model = descriptor()
        ModelAssetURLProtocol.body = payload
        let events = ModelEventCollector()
        let store = RemoteModelStore(directory: root, session: session) { package in
            XCTAssertEqual(try Data(contentsOf: package.appendingPathComponent("Manifest.json")), self.payload)
            return try Self.fakeCompile(package)
        }
        let compiled = try await store.prepare(model, baseURL: baseURL) { events.append($0) }
        XCTAssertTrue(FileManager.default.fileExists(atPath: compiled.path))
        let requests = ModelAssetURLProtocol.requestCount
        XCTAssertEqual(requests, 1)
        let reloaded = RemoteModelStore(directory: root, session: session) { _ in
            XCTFail("Warm cache must not compile again"); throw TestFailure.compile
        }
        let cached = try await reloaded.prepare(model, baseURL: baseURL)
        XCTAssertEqual(cached, compiled)
        XCTAssertEqual(ModelAssetURLProtocol.requestCount, requests)
        XCTAssertTrue(events.values.contains(.compiling))
        XCTAssertEqual(events.values.last, .ready)
        XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath: compiled.deletingLastPathComponent().path)
            .contains("Fixture.mlpackage"), "Source package should be removed after compilation")
    }

    func testChecksumMismatchNeverCompilesOrInstalls() async throws {
        ModelAssetURLProtocol.body = Data(repeating: 0, count: payload.count)
        try await assertRejected(descriptor()) { error in
            guard case ModelDownloadError.checksumMismatch = error else { return XCTFail("\(error)") }
        }
    }

    func testForceDownloadFetchesAndCompilesAnAlreadyInstalledModel() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let session = session(); defer { session.invalidateAndCancel() }
        ModelAssetURLProtocol.body = payload
        let events = ModelEventCollector()
        let store = RemoteModelStore(directory: root, session: session, compiler: { try Self.fakeCompile($0) })
        let model = descriptor()
        let first = try await store.prepare(model, baseURL: baseURL)
        let replaced = try await store.prepare(model, baseURL: baseURL, forceDownload: true) { events.append($0) }
        XCTAssertEqual(first, replaced)
        XCTAssertEqual(ModelAssetURLProtocol.requestCount, 2, "Redownload must bypass the installed-model cache")
        XCTAssertTrue(events.values.contains(.compiling))
        XCTAssertEqual(events.values.last, .ready)
        let cached = try await store.cachedModelURL(for: model)
        XCTAssertEqual(cached, replaced)
    }

    func testFailedRedownloadPreservesTheExistingInstallation() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let session = session(); defer { session.invalidateAndCancel() }
        ModelAssetURLProtocol.body = payload
        let store = RemoteModelStore(directory: root, session: session, compiler: { try Self.fakeCompile($0) })
        let model = descriptor()
        let first = try await store.prepare(model, baseURL: baseURL)
        ModelAssetURLProtocol.body = Data(repeating: 0, count: payload.count)
        do {
            _ = try await store.prepare(model, baseURL: baseURL, forceDownload: true)
            XCTFail("Expected checksum failure")
        } catch ModelDownloadError.checksumMismatch {}
        let cached = try await store.prepare(model, baseURL: baseURL)
        XCTAssertEqual(cached, first)
        XCTAssertEqual(ModelAssetURLProtocol.requestCount, 2, "Old verified installation remains usable offline")
        XCTAssertEqual(try Data(contentsOf: first.appendingPathComponent("coremldata.bin")), Data("compiled".utf8))
        XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath: root.path).contains { $0.hasPrefix(".preparing-") })
    }

    func testReportsByteProgressBeforeDownloadFinishes() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let session = session(); defer { session.invalidateAndCancel() }
        let body = Data(repeating: 7, count: 1_048_576)
        ModelAssetURLProtocol.body = body
        ModelAssetURLProtocol.chunkSize = 65_536
        let model = RemoteModelDescriptor(id: "progress", version: "v1", packageName: "Fixture.mlpackage", files: [
            .init(path: "Manifest.json", asset: "manifest.json", bytes: Int64(body.count),
                  sha256: SHA256.hash(data: body).map { String(format: "%02x", $0) }.joined())
        ])
        let events = ModelEventCollector()
        let store = RemoteModelStore(directory: root, session: session) { package in
            let intermediate = events.values.compactMap { event -> Double? in
                if case .downloading(let fraction) = event, fraction > 0, fraction < 1 { return fraction }
                return nil
            }
            XCTAssertGreaterThan(intermediate.count, 1, "Progress must update during transfer, before compiling")
            XCTAssertEqual(intermediate, intermediate.sorted())
            return try Self.fakeCompile(package)
        }
        _ = try await store.prepare(model, baseURL: baseURL) { events.append($0) }
        XCTAssertTrue(events.values.contains(.downloading(1)))
    }

    func testWrongSizeNeverCompilesOrInstalls() async throws {
        ModelAssetURLProtocol.body = Data("short".utf8)
        try await assertRejected(descriptor()) { error in
            guard case ModelDownloadError.invalidSize = error else { return XCTFail("\(error)") }
        }
    }

    func testCancellationDuringDownloadCleansStagingAndAllowsRetry() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let session = session(); defer { session.invalidateAndCancel() }
        let body = Data(repeating: 7, count: 1_048_576)
        ModelAssetURLProtocol.body = body
        ModelAssetURLProtocol.chunkSize = 65_536
        let model = RemoteModelDescriptor(id: "cancel", version: "v1", packageName: "Fixture.mlpackage", files: [
            .init(path: "Manifest.json", asset: "manifest.json", bytes: Int64(body.count),
                  sha256: SHA256.hash(data: body).map { String(format: "%02x", $0) }.joined())
        ])
        let transferring = expectation(description: "first bytes")
        transferring.assertForOverFulfill = false
        let store = RemoteModelStore(directory: root, session: session, compiler: { try Self.fakeCompile($0) })
        let base = baseURL
        let task = Task {
            try await store.prepare(model, baseURL: base) { event in
                if case .downloading(let fraction) = event, fraction > 0, fraction < 1 {
                    transferring.fulfill()
                }
            }
        }
        await fulfillment(of: [transferring], timeout: 3)
        task.cancel()
        do { _ = try await task.value; XCTFail("Expected cancellation") } catch is CancellationError {}
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: root.path).isEmpty)
        let cached = try await store.cachedModelURL(for: model)
        XCTAssertNil(cached)
        _ = try await store.prepare(model, baseURL: base)
        XCTAssertEqual(ModelAssetURLProtocol.requestCount, 2)
    }

    func testHTTPFailureNeverCompilesOrInstalls() async throws {
        ModelAssetURLProtocol.body = payload; ModelAssetURLProtocol.status = 403
        try await assertRejected(descriptor()) { error in
            guard case ModelDownloadError.httpStatus(403) = error else { return XCTFail("\(error)") }
        }
    }

    func testPathTraversalRejectedBeforeNetworking() async throws {
        try await assertRejected(descriptor(path: "../Manifest.json")) { error in
            guard case ModelDownloadError.invalidDescriptor = error else { return XCTFail("\(error)") }
        }
        XCTAssertEqual(ModelAssetURLProtocol.requestCount, 0)
    }

    func testCompilationFailureCanRetryFromCleanState() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let session = session(); defer { session.invalidateAndCancel() }
        ModelAssetURLProtocol.body = payload
        let failing = RemoteModelStore(directory: root, session: session) { _ in throw TestFailure.compile }
        do { _ = try await failing.prepare(descriptor(), baseURL: baseURL); XCTFail("Expected failure") }
        catch TestFailure.compile {}
        let cache = try await failing.cachedModelURL(for: descriptor())
        XCTAssertNil(cache)
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: root.path).isEmpty)
        let retry = RemoteModelStore(directory: root, session: session, compiler: { try Self.fakeCompile($0) })
        _ = try await retry.prepare(descriptor(), baseURL: baseURL)
        XCTAssertEqual(ModelAssetURLProtocol.requestCount, 2)
    }

    func testCancellationDuringCompilationCleansStaging() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let session = session(); defer { session.invalidateAndCancel() }
        ModelAssetURLProtocol.body = payload
        let compiling = expectation(description: "compiling")
        let store = RemoteModelStore(directory: root, session: session) { _ in
            compiling.fulfill()
            try await Task.sleep(for: .seconds(20))
            throw TestFailure.compile
        }
        let model = descriptor(), base = baseURL
        let task = Task { try await store.prepare(model, baseURL: base) }
        await fulfillment(of: [compiling], timeout: 3)
        task.cancel()
        do { _ = try await task.value; XCTFail("Expected cancellation") } catch is CancellationError {}
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: root.path).isEmpty)
        let cached = try await store.cachedModelURL(for: model)
        XCTAssertNil(cached)
    }

    func testSuccessfulVersionUpdateRemovesOnlyTheSameModelsOldInstallation() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let session = session(); defer { session.invalidateAndCancel() }
        ModelAssetURLProtocol.body = payload
        let store = RemoteModelStore(directory: root, session: session, compiler: { try Self.fakeCompile($0) })
        let first = try await store.prepare(descriptor(), baseURL: baseURL)
        let other = try await store.prepare(descriptor(id: "other"), baseURL: baseURL)
        let updated = Data("updated-manifest".utf8)
        ModelAssetURLProtocol.body = updated
        let second = try await store.prepare(descriptor(version: "v2", contents: updated), baseURL: baseURL)
        XCTAssertNotEqual(first, second)
        XCTAssertFalse(FileManager.default.fileExists(atPath: first.deletingLastPathComponent().path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: second.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: other.path))
        XCTAssertEqual(ModelAssetURLProtocol.requestCount, 3)
    }

    func testFailedVersionUpdateKeepsOldModel() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let session = session(); defer { session.invalidateAndCancel() }
        ModelAssetURLProtocol.body = payload
        let store = RemoteModelStore(directory: root, session: session, compiler: { try Self.fakeCompile($0) })
        let first = try await store.prepare(descriptor(), baseURL: baseURL)
        let updated = Data("updated-manifest".utf8)
        ModelAssetURLProtocol.body = Data(repeating: 0, count: updated.count)
        do {
            _ = try await store.prepare(descriptor(version: "v2", contents: updated), baseURL: baseURL)
            XCTFail("Expected checksum failure")
        } catch ModelDownloadError.checksumMismatch {}
        let cached = try await store.cachedModelURL(for: descriptor())
        XCTAssertEqual(cached, first)
    }

    func testManualRemovalDeletesAllModelsAndAbandonedFilesWithoutDownloading() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let session = session(); defer { session.invalidateAndCancel() }
        ModelAssetURLProtocol.body = payload
        let store = RemoteModelStore(directory: root, session: session, compiler: { try Self.fakeCompile($0) })
        _ = try await store.prepare(descriptor(), baseURL: baseURL)
        _ = try await store.prepare(descriptor(id: "other"), baseURL: baseURL)
        try Data("abandoned".utf8).write(to: root.appendingPathComponent(".preparing-abandoned"))
        let requests = ModelAssetURLProtocol.requestCount
        try await store.removeAll()
        try await store.removeAll()
        let cached = try await store.cachedModelURL(for: descriptor())
        XCTAssertNil(cached)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))
        XCTAssertEqual(ModelAssetURLProtocol.requestCount, requests)
        _ = try await store.prepare(descriptor(), baseURL: baseURL)
        XCTAssertEqual(ModelAssetURLProtocol.requestCount, requests + 1)
    }

    func testCancelledVersionUpdateKeepsOldModel() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let session = session(); defer { session.invalidateAndCancel() }
        ModelAssetURLProtocol.body = payload
        let originalStore = RemoteModelStore(directory: root, session: session, compiler: { try Self.fakeCompile($0) })
        let first = try await originalStore.prepare(descriptor(), baseURL: baseURL)
        let store = RemoteModelStore(directory: root, session: session) { package in
            withUnsafeCurrentTask { $0?.cancel() }
            return try Self.fakeCompile(package)
        }
        let updated = Data("updated-manifest".utf8)
        ModelAssetURLProtocol.body = updated
        do {
            _ = try await store.prepare(descriptor(version: "v2", contents: updated), baseURL: baseURL)
            XCTFail("Expected cancellation")
        } catch is CancellationError {}
        let cached = try await originalStore.cachedModelURL(for: descriptor())
        XCTAssertEqual(cached, first)
        XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath: root.path).contains { $0.hasPrefix(".preparing-") })
    }

    func testReleaseAndAssetChangesReuseExistingContentsWithoutMovingOrDownloading() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let session = session(); defer { session.invalidateAndCancel() }
        ModelAssetURLProtocol.body = payload
        let store = RemoteModelStore(directory: root, session: session, compiler: { try Self.fakeCompile($0) })
        let first = try await store.prepare(descriptor(), baseURL: baseURL)
        // Installation directories are opaque: receipts from earlier releases remain reusable in place.
        let existing = root.appendingPathComponent("previous-installation")
        try FileManager.default.moveItem(at: first.deletingLastPathComponent(), to: existing)
        let original = descriptor()
        let renamed = RemoteModelDescriptor(id: original.id, version: "0.1.0", packageName: "Renamed.mlpackage",
            files: original.files.map { .init(path: $0.path, asset: "renamed.json", bytes: $0.bytes, sha256: $0.sha256) })
        let requests = ModelAssetURLProtocol.requestCount
        let cached = try await store.cachedModelURL(for: renamed)
        XCTAssertEqual(cached, existing.appendingPathComponent("model.mlmodelc", isDirectory: true))
        let prepared = try await store.prepare(renamed, baseURL: URL(string: "https://new.example/0.1.0")!)
        XCTAssertEqual(cached, prepared)
        XCTAssertEqual(ModelAssetURLProtocol.requestCount, requests)
        XCTAssertTrue(FileManager.default.fileExists(atPath: existing.path))
        let receipt = try JSONDecoder().decode(RemoteModelDescriptor.self,
            from: Data(contentsOf: existing.appendingPathComponent("ready.json")))
        XCTAssertEqual(receipt.version, "v1", "Availability must not rewrite or move a valid installation")
    }

    func testChangedContentsInvalidateCacheEvenWithoutAVersionChange() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let session = session(); defer { session.invalidateAndCancel() }
        ModelAssetURLProtocol.body = payload
        let store = RemoteModelStore(directory: root, session: session, compiler: { try Self.fakeCompile($0) })
        let first = try await store.prepare(descriptor(), baseURL: baseURL)
        let updated = Data("updated-manifest".utf8)
        let changed = descriptor(contents: updated)
        let cached = try await store.cachedModelURL(for: changed)
        XCTAssertNil(cached)
        ModelAssetURLProtocol.body = updated
        _ = try await store.prepare(changed, baseURL: baseURL)
        XCTAssertFalse(FileManager.default.fileExists(atPath: first.path))
        XCTAssertEqual(ModelAssetURLProtocol.requestCount, 2)
    }

    func testRemovalDuringCompilationWaitsAndPreventsModelReappearing() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let session = session(); defer { session.invalidateAndCancel() }
        ModelAssetURLProtocol.body = payload
        let compiling = expectation(description: "compiler started")
        let cancelled = expectation(description: "compiler cancelled")
        let gate = CompilationGate()
        let store = RemoteModelStore(directory: root, session: session) { package in
            compiling.fulfill()
            await withTaskCancellationHandler { await gate.wait() } onCancel: { cancelled.fulfill() }
            return try Self.fakeCompile(package)
        }
        let model = descriptor(), base = baseURL
        let installation = Task { try await store.prepare(model, baseURL: base) }
        await fulfillment(of: [compiling], timeout: 3)
        let removal = Task { try await store.removeAll() }
        await fulfillment(of: [cancelled], timeout: 3)
        do {
            _ = try await store.prepare(model, baseURL: base)
            XCTFail("Cannot install while clearing")
        } catch is CancellationError {}
        await gate.release()
        try await removal.value
        do { _ = try await installation.value; XCTFail("Expected cancellation") } catch is CancellationError {}
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))
    }

    func testAvailabilityChecksNeverDownloadAndDetectDeletedModels() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let session = session(); defer { session.invalidateAndCancel() }
        ModelAssetURLProtocol.body = payload
        let model = descriptor()
        let store = RemoteModelStore(directory: root, session: session, compiler: { try Self.fakeCompile($0) })
        let missing = try await store.cachedModelURL(for: model)
        XCTAssertNil(missing)
        XCTAssertEqual(ModelAssetURLProtocol.requestCount, 0)

        let compiled = try await store.prepare(model, baseURL: baseURL)
        let ready = try await store.cachedModelURL(for: model)
        XCTAssertEqual(ready, compiled)
        let requests = ModelAssetURLProtocol.requestCount
        try FileManager.default.removeItem(at: compiled.appendingPathComponent("coremldata.bin"))
        let deletedFile = try await store.cachedModelURL(for: model)
        XCTAssertNil(deletedFile, "A stale receipt must not hide missing model files")
        try FileManager.default.removeItem(at: compiled)
        let deletedDirectory = try await store.cachedModelURL(for: model)
        XCTAssertNil(deletedDirectory)
        XCTAssertEqual(ModelAssetURLProtocol.requestCount, requests, "Checking availability must remain read-only")

        _ = try await store.prepare(model, baseURL: baseURL)
        XCTAssertEqual(ModelAssetURLProtocol.requestCount, requests + 1, "Explicit download repairs the missing model")
    }

    private func assertRejected(_ model: RemoteModelDescriptor, check: (Error) -> Void) async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let session = session(); defer { session.invalidateAndCancel() }
        let store = RemoteModelStore(directory: root, session: session) { _ in
            XCTFail("Invalid download must not reach compiler"); throw TestFailure.compile
        }
        do { _ = try await store.prepare(model, baseURL: baseURL); XCTFail("Expected rejection") }
        catch { check(error) }
        let cached = try await store.cachedModelURL(for: model)
        XCTAssertNil(cached)
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: root.path).isEmpty)
    }

    private func descriptor(path: String = "Manifest.json", version: String = "v1", id: String = "fixture",
                            contents: Data? = nil) -> RemoteModelDescriptor {
        let data = contents ?? payload
        return RemoteModelDescriptor(id: id, version: version, packageName: "Fixture.mlpackage", files: [
            .init(path: path, asset: "manifest.json", bytes: Int64(data.count),
                  sha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
        ])
    }

    private func directory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [ModelAssetURLProtocol.self]
        return URLSession(configuration: config)
    }

    private static func fakeCompile(_ package: URL) throws -> URL {
        let output = package.deletingLastPathComponent().appendingPathComponent(UUID().uuidString + ".mlmodelc")
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try Data("compiled".utf8).write(to: output.appendingPathComponent("coremldata.bin"))
        return output
    }

    private enum TestFailure: Error { case compile }
}

private actor CompilationGate {
    private var continuation: CheckedContinuation<Void, Never>?
    private var released = false
    func wait() async {
        if released { return }
        await withCheckedContinuation { continuation = $0 }
    }
    func release() {
        released = true
        continuation?.resume()
        continuation = nil
    }
}

private final class ModelEventCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [ModelPreparationProgress] = []
    func append(_ event: ModelPreparationProgress) { lock.lock(); events.append(event); lock.unlock() }
    var values: [ModelPreparationProgress] { lock.lock(); defer { lock.unlock() }; return events }
}

private final class ModelAssetURLProtocol: URLProtocol {
    static var body = Data()
    static var status = 200
    static var requestCount = 0
    static var chunkSize: Int?
    static func reset() { body = Data(); status = 200; requestCount = 0; chunkSize = nil }
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        Self.requestCount += 1
        let response = HTTPURLResponse(url: request.url!, statusCode: Self.status, httpVersion: nil,
                                       headerFields: ["Content-Length": "\(Self.body.count)"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        let body = Self.body
        if let size = Self.chunkSize {
            for offset in stride(from: 0, to: body.count, by: size) {
                client?.urlProtocol(self, didLoad: body.subdata(in: offset..<min(offset + size, body.count)))
                Thread.sleep(forTimeInterval: 0.03)
            }
        } else {
            client?.urlProtocol(self, didLoad: body)
        }
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}
