# Short compressed tail

`short-tail.mp3` is an original generated 440 Hz stereo sine, with 417 bytes
removed from its end. Its Xing header still advertises 578,724 frames while
Apple's decoder returns 577,728 frames. It reproduces a playable compressed file
whose last read is shorter than the advertised length. No music recording is used.

Generated with FFmpeg 8 / libmp3lame:

```sh
ffmpeg -f lavfi -i 'sine=frequency=440:sample_rate=44100:duration=13.123' \
  -ac 2 -codec:a libmp3lame -q:a 2 full.mp3
python3 -c "from pathlib import Path; Path('short-tail.mp3').write_bytes(Path('full.mp3').read_bytes()[:-417])"
```

The fixture and tests are provided under Apache-2.0 with the package.
