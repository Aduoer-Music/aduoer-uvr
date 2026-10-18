#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage immutable GitHub Release assets without checking weights into Git."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", default="0.1.0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = f"https://github.com/Aduoer-Music/aduoer-uvr/releases/download/{args.tag}"
    catalog = {"schemaVersion": 1, "release": args.tag, "baseURL": base, "models": []}
    for identifier, name in [("mdx9482", "UVR9482FP16"), ("instHQ3", "UVRInstHQ3FP16")]:
        package = args.model_dir / f"{name}.mlpackage"
        model = {"id": identifier, "version": args.tag, "packageName": package.name, "files": []}
        for relative, suffix in [("Manifest.json", "manifest.json"),
                                 ("Data/com.apple.CoreML/model.mlmodel", "model.mlmodel"),
                                 ("Data/com.apple.CoreML/weights/weight.bin", "weights.bin")]:
            source = package / relative
            asset = f"{name}-{suffix}"
            shutil.copyfile(source, args.output / asset)
            model["files"].append({"path": relative, "asset": asset,
                                   "bytes": source.stat().st_size,
                                   "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
        catalog["models"].append(model)
    (args.output / "models.json").write_text(json.dumps(catalog, indent=2) + "\n")
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(root / "models" / "MODEL_NOTICES.md", args.output / "MODEL_NOTICES.md")
    print(json.dumps({"output": str(args.output), "release": args.tag,
                      "assetBytes": sum(p.stat().st_size for p in args.output.iterdir())}))


if __name__ == "__main__":
    main()
