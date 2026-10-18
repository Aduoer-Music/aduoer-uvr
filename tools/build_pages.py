#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the public listening site from README and its sample download table."""
import html
from pathlib import Path
import re
import shutil

import markdown

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/Aduoer-Music/aduoer-uvr"
SITE = "https://aduoer-music.github.io/aduoer-uvr/"


def listening_cards(match):
    lines = match.group(0).strip().splitlines()
    labels = [cell.strip() for cell in lines[0].strip("|").split("|")][1:]
    sections = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        song = html.escape(cells[0])
        tracks = []
        for label, cell in zip(labels, cells[1:]):
            source = re.fullmatch(r"\[下载\]\((samples/[^)]+\.mp3)\?raw=true\)", cell)
            if source is None or not (ROOT / source[1]).is_file():
                raise ValueError("Missing listening sample")
            name, url = html.escape(label), html.escape(source[1], quote=True)
            tracks.append(
                f'<div class="track"><span>{name}</span>'
                f'<audio controls preload="none" aria-label="{song} · {name}" src="{url}"></audio>'
                f'<a href="{url}" download aria-label="下载 {song} · {name}">下载</a></div>'
            )
        sections.append(
            f'<section class="song"><div class="song-heading"><h3>{song}</h3>'
            '<button type="button" data-restart>回到开头</button></div>'
            + "".join(tracks) + "</section>"
        )
    return '<div class="listening-grid">' + "".join(sections) + "</div>\n"


def main():
    source = (ROOT / "README.md").read_text()
    source = source.replace(
        f"[打开在线试听页面]({SITE})，直接播放原曲、9482 与 Inst_HQ_3 的人声和伴奏。",
        "直接播放原曲、9482 与 Inst_HQ_3 的人声和伴奏。同一首歌切换音轨时沿用当前进度。",
    ).replace("### 独立音轨下载", "### 对照试听")
    source, count = re.subn(
        r"\| 歌曲 \| 原曲 \|[^\n]*\n\|[-|]+\|\n(?:\|[^\n]+\|\n)+",
        listening_cards, source,
    )
    if count != 1:
        raise ValueError("Expected exactly one README listening table")
    # Documentation and licenses stay linked to their repository files.
    source = re.sub(
        r"\]\((?!https?://|#)([^)]+)\)",
        lambda match: f"]({REPOSITORY}/blob/main/{match[1]})", source,
    )
    rendered = markdown.markdown(
        source, extensions=["tables", "fenced_code", "toc"],
        extension_configs={"toc": {"slugify": lambda value, separator: re.sub(
            r"[^\w]+", separator, value.lower()).strip(separator)}},
    )
    template = (ROOT / "site/template.html").read_text()
    output = ROOT / "_site"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(exist_ok=True)
    samples = output / "samples"
    samples.mkdir(exist_ok=True)
    for sample in (ROOT / "samples").glob("*.mp3"):
        shutil.copy2(sample, samples / sample.name)
    (output / "index.html").write_text(template.replace("{{CONTENT}}", rendered))
    print("Built README page with 10 audio players in _site/")


if __name__ == "__main__":
    main()
