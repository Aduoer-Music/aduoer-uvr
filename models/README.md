# 模型

支持以下模型，使用 `tools/download_dependencies.py` 下载，再按项目 README 转换：

| 配置 | 上游模型 | 生成文件 |
|---|---|---|
| `mdx9482` | UVR_MDXNET_9482 | `coreml/UVR9482FP16.mlpackage` |
| `instHQ3` | UVR-MDX-NET-Inst_HQ_3 | `coreml/UVRInstHQ3FP16.mlpackage` |

下载来源和 SHA-256 固定在 `../dependencies.lock.json`。

已转换的 FP16 模型发布在 [0.1.0](https://github.com/Aduoer-Music/aduoer-uvr/releases/tag/0.1.0)。下载后的每个文件必须按 [固定清单](releases/0.1.0.json) 校验大小和 SHA-256，再还原 `.mlpackage` 并在使用设备上编译。可使用 Swift 包的 `RemoteModelStore`，由用户手动触发下载，后续离线读取。

上游：[TRvlvr/model_repo](https://github.com/TRvlvr/model_repo)、[sherpa-onnx 模型发布页](https://github.com/k2-fsa/sherpa-onnx/releases/tag/source-separation-models)。参数来源：[UVR 模型配置](https://github.com/TRvlvr/application_data/blob/main/mdx_model_data/model_data.json)。

本项目的 Apache 2.0 许可不覆盖第三方权重。目前未核实这些权重的独立再分发许可；原始权重与转换结果均不进入 Git，Release 资产保留独立来源和许可说明。使用和分发权重前，请核对原始发布者的授权条款；sherpa-onnx 的代码许可不能替代模型授权。详见 [MODEL_NOTICES.md](MODEL_NOTICES.md)。
