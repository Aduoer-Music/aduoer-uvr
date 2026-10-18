#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare supported ONNX models using their catalog configuration."""
import json
from pathlib import Path
import onnx

ROOT = Path(__file__).resolve().parents[1]


def main():
    for config in json.loads((ROOT / "models/catalog.json").read_text()).values():
        original = ROOT / "models/onnx" / (config["name"] + ".onnx")
        target = original.parent / config["source"]
        model = onnx.load(original)
        graph = model.graph.SerializeToString()
        for prop in model.metadata_props:
            if prop.key in ("n_fft", "win_length"):
                prop.value = str(config["fft"])
        if target != original:
            onnx.save(model, target)
        assert onnx.load(target).graph.SerializeToString() == graph
        print("Prepared", target.name)


if __name__ == "__main__":
    main()
