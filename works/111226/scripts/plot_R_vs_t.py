#!/usr/bin/env python3
"""Plot frozen selection-derived cutoff R against diffusion timestep t.

Reads selected_cutoffs.csv only. Does not run FFT, select new cutoffs,
fit a curve, or calculate a correlation coefficient.

Requires matplotlib; other imports are from the Python standard library.
"""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', type=Path, required=True,
                        help='Path to selection/selected_cutoffs.csv')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output PNG path')
    args = parser.parse_args()
    if args.output.suffix.lower() != '.png':
        parser.error('--output must end in .png')
    with args.input_csv.open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'timestep', 'candidate', 'radius', 'n_tied_best'}
        if not required.issubset(reader.fieldnames or []):
            parser.error('CSV must contain: ' + ', '.join(sorted(required)))
        rows = list(reader)
    if not rows:
        parser.error('CSV has no data rows')

    values = []
    for row in rows:
        t = int(row['timestep'])
        if row['candidate'] == 'full' or not row['radius'].strip():
            parser.error('A selected full-spectrum option has no single numeric R. '
                         'Use a separate categorical panel for this dataset.')
        radius = float(row['radius'])
        ties = int(row['n_tied_best'])
        if t < 0 or not math.isfinite(radius) or radius <= 0 or ties < 1:
            parser.error(f'Invalid timestep, radius or tie count at t={t}')
        values.append((t, radius, ties))
    values.sort()
    if len({t for t, _, _ in values}) != len(values):
        parser.error('CSV contains duplicate timesteps')

    ts, radii, ties = zip(*values)
    blue = '#1764A4'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})
    fig, ax = plt.subplots(figsize=(12, 7.5))
    fig.subplots_adjust(left=.10, right=.97, top=.81, bottom=.28)
    ax.plot(ts, radii, color=blue, lw=1.2, alpha=.75, zorder=2)
    for is_tie in (False, True):
        points = [(t, r) for t, r, n in values if (n > 1) == is_tie]
        if points:
            x, y = zip(*points)
            ax.scatter(x, y, s=31, facecolors='white' if is_tie else blue,
                       edgecolors=blue, linewidths=1.4, zorder=3)
    ax.set_xlabel('Diffusion timestep t', labelpad=10)
    ax.set_ylabel('Selected cutoff R\n(cycles per prepared-image pixel)', labelpad=10)
    ax.set_xlim(min(ts) - 8, max(ts) + 8)
    ax.set_ylim(0, max(radii) * 1.16)
    if min(ts) == 0 and max(ts) == 999:
        ax.set_xticks([0, 100, 250, 500, 750, 900, 999])
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.18)
    handles = [
        Line2D([], [], marker='o', ls='none', color=blue,
               label='Unique highest selection accuracy'),
        Line2D([], [], marker='o', ls='none', color=blue, markerfacecolor='white',
               label='Tied highest accuracy; smallest R chosen'),
    ]
    ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=10)
    fig.text(.10, .94, 'Selected Fourier cutoff R versus diffusion timestep t',
             fontsize=19, weight='bold')
    fig.text(.10, .887,
             f'{len(values)} measured timestep choices | {sum(n > 1 for n in ties)} timesteps with tied-best cutoffs',
             fontsize=12, color='#555555')
    fig.text(.10, .09,
             'Cutoffs come from the selection images; evaluation results do not change these choices.\n'
             'Lines connect sampled points only: no smoothing or fitted relationship. Near-noise winners may be unstable.\n'
             f'Source: {args.input_csv.name}. Only the chosen R is shown; other tied-best cutoffs are omitted.',
             fontsize=10, color='#555555', linespacing=1.6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, facecolor='white')
    plt.close(fig)
    print(f'Saved: {args.output.resolve()}')
    print(f'Plotted {len(values)} existing choices. No analysis rerun or cutoff changes.')


if __name__ == '__main__':
    main()
