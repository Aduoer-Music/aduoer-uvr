#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Download pinned ONNX weights; weights retain the original publisher's terms."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['all', 'mdx9482', 'instHQ3'], default='all')
    args = parser.parse_args()
    catalog = json.loads((ROOT / 'models/catalog.json').read_text())
    selected = {v['name']+'.onnx' for k, v in catalog.items() if args.model in ('all', k)}
    for item in json.loads((ROOT / 'dependencies.lock.json').read_text()):
        if item['name'] not in selected:
            continue
        directory = ROOT / 'models/onnx'; directory.mkdir(parents=True, exist_ok=True)
        target = directory / item['name']
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == item['sha256']:
            print('Verified', target.name); continue
        pending = target.with_suffix('.onnx.pending')
        urllib.request.urlretrieve(item['url'], pending)
        if hashlib.sha256(pending.read_bytes()).hexdigest() != item['sha256']:
            pending.unlink(); raise RuntimeError('Checksum mismatch: '+item['name'])
        pending.replace(target)
        print('Verified', target.name)


if __name__ == '__main__':
    main()
