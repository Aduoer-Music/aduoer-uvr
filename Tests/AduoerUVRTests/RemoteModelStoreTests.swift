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

    func testChangedVersionUsesIndependentInstallation() async throws {
        let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
        let session = session(); defer { session.invalidateAndCancel() }
        ModelAssetURLProtocol.body = payload
        let store = RemoteModelStore(directory: root, session: session, compiler: { try Self.fakeCompile($0) })
        let first = try await store.prepare(descriptor(), baseURL: baseURL)
        let second = try await store.prepare(descriptor(version: "v2"), baseURL: baseURL)
        XCTAssertNotEqual(first, second)
        XCTAssertTrue(FileManager.default.fileExists(atPath: first.path))
        XCTAssertEqual(ModelAssetURLProtocol.requestCount, 2)
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

    private func descriptor(path: String = "Manifest.json", version: String = "v1") -> RemoteModelDescriptor {
        RemoteModelDescriptor(id: "fixture", version: version, packageName: "Fixture.mlpackage", files: [
            .init(path: path, asset: "manifest.json", bytes: Int64(payload.count),
                  sha256: SHA256.hash(data: payload).map { String(format: "%02x", $0) }.joined())
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
