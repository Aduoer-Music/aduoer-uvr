# Aduoer UVR

[Apache-2.0](LICENSE) · iOS 17+ · macOS 14+

在 Apple 设备上本地分离人声和伴奏的 Swift 包，支持 `UVR_MDXNET_9482` 和 `UVR-MDX-NET-Inst_HQ_3`，无需服务器。

## 真机对比

iPhone 17 / iOS 27，每个案例各测试一次。耗时包含解码、模型加载、分离和写盘；内存为整个 App 的峰值 physical footprint。以下结果仅供参考，实际表现会随设备和运行状态变化。

| 模型 / 运行方式 | 歌曲 | 歌曲时长 | 处理耗时 | 峰值内存 |
|---|---|---:|---:|---:|
| 9482 · ONNX CPU | Round We Go | 164.0 秒 | 74.01 秒 | 1186.6 MiB |
| 9482 · ONNX Core ML | Round We Go | 164.0 秒 | 29.73 秒 | 2783.1 MiB |
| 9482 · 原生 Core ML FP16 | Round We Go | 164.0 秒 | 4.47 秒 | 335.6 MiB |
| 9482 · 原生 Core ML FP16 | IF YOU | 264.2 秒 | 4.87 秒 | 332.8 MiB |
| Inst_HQ_3 · 原生 Core ML FP16 | Round We Go | 164.0 秒 | 8.02 秒 | 513.4 MiB |
| Inst_HQ_3 · 原生 Core ML FP16 | IF YOU | 264.2 秒 | 10.65 秒 | 514.2 MiB |

9482 的处理速度更快、内存占用更低；Inst_HQ_3 可作为另一种伴奏效果选择，请结合下方试听判断。

## 试听

[打开在线试听页面](https://aduoer-music.github.io/aduoer-uvr/)，直接播放原曲、9482 与 Inst_HQ_3 的人声和伴奏。

每组使用同一首歌 01:00–01:20 的片段，统一降低音量以留出播放余量。

### 独立音轨下载

| 歌曲 | 原曲 | 9482 人声 | 9482 伴奏 | Inst_HQ_3 人声 | Inst_HQ_3 伴奏 |
|---|---|---|---|---|---|
| Round We Go | [下载](samples/round-we-go-original.mp3?raw=true) | [下载](samples/round-we-go-9482-vocals.mp3?raw=true) | [下载](samples/round-we-go-9482-instrumental.mp3?raw=true) | [下载](samples/round-we-go-hq3-vocals.mp3?raw=true) | [下载](samples/round-we-go-hq3-instrumental.mp3?raw=true) |
| IF YOU | [下载](samples/if-you-original.mp3?raw=true) | [下载](samples/if-you-9482-vocals.mp3?raw=true) | [下载](samples/if-you-9482-instrumental.mp3?raw=true) | [下载](samples/if-you-hq3-vocals.mp3?raw=true) | [下载](samples/if-you-hq3-instrumental.mp3?raw=true) |

试听音频片段及原曲版权归各自权利人，不适用本项目的 Apache 2.0 代码许可。

## 接入

将本仓库添加为 Swift Package 依赖，链接 `AduoerUVR` 产品。将本地生成的 `.mlpackage` 加入 App 资源，由 Xcode 编译。

```swift
import AduoerUVR

let model = Bundle.main.url(forResource: "UVRInstHQ3FP16", withExtension: "mlmodelc")!
let result = try await SourceSeparator().separate(
    input: localAudioURL,
    directory: workingDirectory,
    modelURL: model,
    configuration: .instHQ3,
    progress: { fraction in /* update UI on MainActor */ }
)
// result.vocals / result.instrumental：44.1kHz 双声道 WAV
```

使用 9482 时，将模型名换为 `UVR9482FP16`，配置换为 `.mdx9482`。每次任务使用独立工作目录，失败或取消后由调用方清理该目录。

压缩音频先使用固定小缓冲顺序解码为临时 PCM CAF，再按实际帧数分块推理，避免文件头长度估算偏差导致尾块失败。临时文件占用磁盘空间随歌曲时长增长，解码和推理缓冲区的大小固定。

## 准备模型

在安装 Xcode 的 Apple Silicon Mac 上，使用 Python 3.9：

```sh
python3.9 -m venv .native-venv
.native-venv/bin/pip install -r requirements-native.txt
.native-venv/bin/python tools/download_dependencies.py
.native-venv/bin/python tools/prepare_models.py
.native-venv/bin/python tools/convert_coreml.py --model mdx9482
.native-venv/bin/python tools/convert_coreml.py --model instHQ3
```

模型生成在 `models/coreml/`。下载来源和校验和见 `dependencies.lock.json`；权重许可见 [模型说明](models/README.md)。

### 下载已转换的模型

固定版本：[Core ML FP16 v1](https://github.com/Aduoer-Music/aduoer-uvr/releases/tag/coreml-fp16-v1)。9482 约 15 MB，Inst_HQ_3 约 33.5 MB。Release 中的 `models.json` 记录每个文件的路径、大小与 SHA-256；相同清单保存在 [models/releases/coreml-fp16-v1.json](models/releases/coreml-fp16-v1.json)。权重不受本项目 Apache 2.0 代码许可覆盖，详见 [模型说明](models/MODEL_NOTICES.md)。

`RemoteModelStore` 逐文件下载并校验，重建 `.mlpackage`，在当前设备编译后原子保存。模型文件流式写盘，编译成功后删除下载源包；完成后无需联网即可复用。请在用户明确点击下载后调用 `prepare`，选择模型和进入演唱模式只检查 `cachedModelURL`。

```swift
let catalog = try JSONDecoder().decode(ModelReleaseCatalog.self, from: pinnedCatalogData)
let descriptor = catalog.models.first { $0.id == "mdx9482" }!
let store = RemoteModelStore(directory: URL.applicationSupportDirectory.appendingPathComponent("UVRModels"))
// Only from an explicit download action:
let modelURL = try await store.prepare(descriptor, baseURL: catalog.baseURL) { progress in
    // .downloading(fraction), .compiling, .ready
}
// Later, offline; returns nil if the selected model has not been downloaded:
let installedURL = try await store.cachedModelURL(for: descriptor)
```

客户端应内置经过审核的固定清单，不能直接信任网络清单来替换校验值。发布者可用 `tools/prepare_model_release.py` 生成资产，更新模型时使用新的版本标签，不替换旧版文件。

用户手动选择重新下载时，可传入 `prepare(descriptor, baseURL: catalog.baseURL, forceDownload: true)`，即使已有安装也会重新获取、校验并编译。下载或校验失败时保留原有安装。

## 测试

```sh
swift test -c release

# 完成模型转换后，可运行完整本地测试
UVR_TEST_MODEL_DIR="$PWD/models/coreml" \
UVR_ORACLE_DIR="$PWD/.deps/oracles" \
swift test -c release
```

未提供模型时，相关测试会跳过。测试音频由代码生成。

## 许可

原创代码和测试使用 [Apache 2.0](LICENSE)。第三方依赖、模型和试听音频保留各自许可，详见 [第三方说明](THIRD_PARTY_NOTICES.md)。
