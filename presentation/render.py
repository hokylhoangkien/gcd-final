"""Render the single Markdown presentation to a vector PDF; no benchmark figures."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties


def wrap_line(text, renderer, size, max_pixels):
    font = FontProperties(family='DejaVu Sans', size=size)
    width = lambda s: renderer.get_text_width_height_descent(s, font, False)[0]
    lines, current = [], ''
    for word in text.split():
        trial = (current+' '+word).strip()
        if width(trial) <= max_pixels:
            current = trial
            continue
        if current:
            lines.append(current)
            current = ''
        # Long hashes/URLs must also fit without overflowing the page.
        for char in word:
            if current and width(current+char) > max_pixels:
                lines.append(current)
                current = ''
            current += char
    if current:
        lines.append(current)
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).with_name('CRISH_GCD.md'))
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('CRISH_GCD.pdf'))
    parser.add_argument('--preview', type=Path, help='Optional preview directory outside presentation/')
    args = parser.parse_args()
    slides = [part.strip() for part in args.source.read_text().split('\n---\n') if part.strip()]
    checks = []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.usetex': False})
    metadata = {'Title': 'CRiSH-GCD v2.1 — Research and implementation handoff',
                'Author': 'CRiSH-GCD research project',
                'CreationDate': datetime(2026, 11, 4, tzinfo=timezone.utc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output, metadata=metadata) as pdf:
        for number, slide in enumerate(slides, 1):
            lines = slide.splitlines()
            title = lines[0].removeprefix('# ')
            body = [line.removeprefix('- ').removeprefix('## ') for line in lines[1:] if line.strip()]
            fig = plt.figure(figsize=(13.3333, 7.5), dpi=120, facecolor='#f4f6f8')
            fig.text(.052, .93, 'CRiSH / GCD', color='#087f8c', fontsize=12, fontweight='bold')
            fig.text(.052, .84, title, fontsize=29 if len(title) > 45 else 34,
                     color='#132b44', fontweight='bold')
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            size = 19.
            while True:
                wrapped = [wrap_line(item, renderer, size, fig.bbox.width*.855) for item in body]
                height = sum(len(item)*size*1.32/72 for item in wrapped)+max(0, len(body)-1)*.15
                if height <= 4.98 or size <= 15:
                    break
                size -= .5
            if height > 4.98:
                raise ValueError(f'Slide {number} too dense: split its content in the Markdown source')
            y = 5.8
            for item in wrapped:
                fig.text(.054, y/7.5, '•', fontsize=size, color='#087f8c', va='top')
                fig.text(.077, y/7.5, '\n'.join(item), fontsize=size, linespacing=1.32,
                         color='#243b53', va='top')
                y -= len(item)*size*1.32/72+.15
            fig.text(.052, .035, 'RESEARCH IMPLEMENTATION  ·  CPU VALIDATION  ·  BENCHMARK NOT RUN',
                     color='#526d82', fontsize=9)
            fig.text(.95, .035, f'{number:02d} / {len(slides):02d}', ha='right', fontsize=10, color='#526d82')
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            if args.preview:
                args.preview.mkdir(parents=True, exist_ok=True)
                fig.savefig(args.preview/f'slide_{number:02d}.png', dpi=100, facecolor=fig.get_facecolor())
            checks.append(dict(slide=number, font_size=size, body_height_inches=round(height, 3)))
            plt.close(fig)
    print(json.dumps(dict(output=str(args.output), slides=len(slides), layout=checks), indent=2))


if __name__ == '__main__':
    main()
