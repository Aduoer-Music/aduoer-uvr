# Core ML model assets

These are FP16 Core ML conversions of third-party UVR MDX weights:

| Core ML model | Original weight | Original SHA-256 |
| --- | --- | --- |
| UVR9482FP16 | UVR_MDXNET_9482.onnx | 9d78f8566fa8198065214ab628be1de966a500c57786695aa4b13e2b27a7727d |
| UVRInstHQ3FP16 | UVR-MDX-NET-Inst_HQ_3.onnx | c207b845d51dc1303c543503f6f5a8b596da5d7de97bead77b3e16ec38e8d5b3 |

Original downloads: [sherpa-onnx source-separation models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/source-separation-models).
Model upstream: [TRvlvr/model_repo](https://github.com/TRvlvr/model_repo).
Conversion source: [tools/convert_coreml.py](https://github.com/Aduoer-Music/aduoer-uvr/blob/main/tools/convert_coreml.py).

The project's Apache-2.0 license applies to its original code, not these weights.
The original models retain their original ownership and terms. An independent
redistribution license has not been verified; publishing these conversions does
not establish an Apache-2.0 grant for the model weights.

`models.json` records the SHA-256, byte count, package path and Release asset name
for each file. Reconstruct the indicated `.mlpackage` directory and compile it
on the target device with Core ML. No inference audio, access credentials or
private App data are included in these assets.
