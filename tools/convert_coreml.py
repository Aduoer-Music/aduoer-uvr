#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert pinned MDX weights to a fixed-shape FP16 ML Program and validate them."""
import argparse
import hashlib
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import onnx
import onnxruntime as ort
from onnx2torch import convert
import torch

ROOT = Path(__file__).resolve().parents[1]


def metrics(actual, expected):
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    return dict(relative_rms=float(np.sqrt(np.mean(delta**2) / max(np.mean(expected.astype(np.float64)**2), 1e-20))),
                error_rms=float(np.sqrt(np.mean(delta**2))),
                reference_rms=float(np.sqrt(np.mean(expected.astype(np.float64)**2))),
                max_abs=float(np.max(np.abs(delta))), finite=bool(np.isfinite(actual).all()))


def spectrogram(pcm, config):
    fft, hop, bins, times = (config[k] for k in ('fft', 'hop', 'bins', 'time_frames'))
    padded = np.zeros((2, hop * (times - 1)), dtype=np.float32)
    padded[:, fft//2:fft//2+pcm.shape[1]] = pcm
    spectrum = torch.stft(torch.from_numpy(padded), n_fft=fft, hop_length=hop,
                          window=torch.hann_window(fft), center=True, return_complex=True)
    return np.stack([spectrum[0].real[:bins].numpy(), spectrum[0].imag[:bins].numpy(),
                     spectrum[1].real[:bins].numpy(), spectrum[1].imag[:bins].numpy()])[None].copy()


def waveform(spectrum, config, frames):
    full = np.zeros((2, config['fft']//2+1, config['time_frames']), dtype=np.complex64)
    full[0, :config['bins']] = spectrum[0, 0] + 1j*spectrum[0, 1]
    full[1, :config['bins']] = spectrum[0, 2] + 1j*spectrum[0, 3]
    primary = torch.istft(torch.from_numpy(full), n_fft=config['fft'], hop_length=config['hop'],
                          window=torch.hann_window(config['fft']), center=True).numpy()
    return primary[:, config['fft']//2:config['fft']//2+frames].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['mdx9482', 'instHQ3'], required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / 'models/catalog.json').read_text())[args.model]
    source = ROOT / 'models/onnx' / config['source']
    target = ROOT / 'models/coreml' / config['package']
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    shape = (1, 4, config['bins'], config['time_frames'])
    fixture = (np.random.default_rng(9482).standard_normal(shape) * 0.3).astype(np.float32)
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(source), sess_options=options, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    expected = session.run(None, {input_name: fixture})[0]
    model = convert(onnx.load(str(source))).eval()
    with torch.no_grad():
        torch_error = metrics(model(torch.from_numpy(fixture)).numpy(), expected)
        print('Torch / ONNX', torch_error, flush=True)
        assert torch_error['finite'] and torch_error['relative_rms'] < 1e-4
        traced = torch.jit.trace(model, torch.from_numpy(fixture))
    converted = ct.convert(traced, convert_to='mlprogram', minimum_deployment_target=ct.target.iOS17,
                           inputs=[ct.TensorType(name='spectrogram', shape=shape, dtype=np.float32)],
                           outputs=[ct.TensorType(name=config['output_name'], dtype=np.float32)],
                           compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.ALL)
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    converted.short_description = config['name'] + ', FP16 compute, fixed batch 1'
    converted.user_defined_metadata.update(source_sha256=checksum, primary_stem=config['primary_stem'],
                                           fft=str(config['fft']), weight_license='See original weight publisher terms')
    converted.save(str(target))
    predicted = converted.predict({'spectrogram': fixture})[config['output_name']]
    fp16_error = metrics(predicted, expected)
    print('FP16 / ONNX', fp16_error, flush=True)
    assert fp16_error['finite'] and fp16_error['relative_rms'] < 0.02
    # Generated audio is deterministic and requires no external media.
    time_axis = np.arange(220500, dtype=np.float32) / 44100
    pcm = np.stack([0.3*np.sin(2*np.pi*440*time_axis), 0.25*np.sin(2*np.pi*523.25*time_axis)])
    x = spectrogram(pcm, config)
    y = session.run(None, {input_name: x})[0]
    predicted_audio = converted.predict({'spectrogram': x})[config['output_name']]
    music_error = metrics(predicted_audio, y)
    print('Audio FP16 / ONNX', music_error, flush=True)
    assert music_error['finite'] and music_error['relative_rms'] < 0.02
    primary = waveform(y, config, pcm.shape[1])
    waveform_error = metrics(waveform(predicted_audio, config, pcm.shape[1]), primary)
    input_rms = float(np.sqrt(np.mean(pcm.astype(np.float64)**2)))
    # Same mixed error budget as Swift's oracle tests; preserve the 2% relative component.
    waveform_error['limit'] = 0.02 * waveform_error['reference_rms'] + 0.0002 * input_rms
    print('Waveform FP16 / ONNX', waveform_error, flush=True)
    assert waveform_error['finite'] and waveform_error['error_rms'] <= waveform_error['limit']
    oracle = ROOT / '.deps/oracles' / args.model; oracle.mkdir(parents=True, exist_ok=True)
    pcm.tofile(oracle / 'input.f32'); x.tofile(oracle / 'spectrum.f32'); y.tofile(oracle / 'output-spectrum.f32')
    primary.tofile(oracle / 'primary.f32')
    print(target, flush=True)


if __name__ == '__main__':
    main()
