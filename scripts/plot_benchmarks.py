"""Validate raw/summary consistency, then chart successful measured cases only."""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ALGORITHMS = ['bitonic-global', 'bitonic-shared', 'radix-1', 'radix-2', 'radix-4', 'radix-8', 'cub']

def key(row):
    return tuple(row[name] for name in ['algorithm', 'key_count', 'distribution', 'seed', 'gpu'])

def load_validated(directory):
    with (directory / 'summary.csv').open(newline='', encoding='utf-8') as f:
        summary = list(csv.DictReader(f))
    with (directory / 'raw.csv').open(newline='', encoding='utf-8') as f:
        raw = list(csv.DictReader(f))
    groups = defaultdict(list)
    for row in raw:
        if row['status'] == 'PASS': groups[key(row)].append(row)
    passed, seen = [], set()
    for row in summary:
        if key(row) in seen: raise ValueError('Duplicate summary case')
        seen.add(key(row))
        if row['status'] != 'PASS': continue
        trials = groups[key(row)]
        if len(trials) != int(row['trials']) or len(trials) != int(row['completed_trials']):
            raise ValueError('Missing raw trials: ' + str(key(row)))
        if len({t['trial'] for t in trials}) != len(trials): raise ValueError('Duplicate trial')
        if row['host_memory'] != 'pinned' or row['build_configuration'] != 'Release':
            raise ValueError('Measurements require pinned memory and Release')
        for field in ['h2d', 'sort', 'd2h', 'end_to_end']:
            values = [float(t[field + '_ms']) for t in trials]
            if any(not math.isfinite(v) or v < 0 for v in values): raise ValueError('Invalid trial time')
            expected = statistics.median(values)
            actual = float(row['median_' + field + '_ms'])
            if not math.isclose(expected, actual, rel_tol=1e-7, abs_tol=1e-9): raise ValueError('Median mismatch')
        sort_ms = float(row['median_sort_ms'])
        if sort_ms <= 0: continue  # Below event resolution; no performance comparison.
        throughput = int(row['key_count']) * 1000 / sort_ms
        if not math.isclose(throughput, float(row['keys_per_second']), rel_tol=1e-7): raise ValueError('Throughput mismatch')
        passed.append(row)
    cub = {key(r)[1:]: r for r in passed if r['algorithm'] == 'cub'}
    for row in passed:
        baseline = cub.get(key(row)[1:])
        if baseline and int(row['key_count']) > 0:
            expected = 100 * float(row['keys_per_second']) / float(baseline['keys_per_second'])
            if not row['percentage_of_cub'] or not math.isclose(float(row['percentage_of_cub']), expected, rel_tol=1e-7):
                raise ValueError('CUB percentage mismatch')
        elif row['percentage_of_cub']:
            raise ValueError('CUB percentage without successful matching baseline')
    return passed

def generate(directory, output, fixture=False):
    rows = load_validated(directory)
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('data_kind') != 'measured' and not fixture:
        raise ValueError('Non-measured input requires --fixture')
    if metadata.get('data_kind') != 'measured': fixture = True
    if not rows: raise ValueError('No successful measurable cases')
    output.mkdir(parents=True, exist_ok=True)
    label = 'SYNTHETIC FIXTURE - NOT GPU MEASUREMENTS' if fixture else 'Measured GPU results'
    grouped = defaultdict(list)
    for row in rows: grouped[(row['gpu'], row['seed'])].append(row)
    notes = [label, 'Performance comparisons exclude failed/skipped and unresolved event times.']
    for index, ((gpu, seed), group) in enumerate(grouped.items()):
        distributions = sorted({r['distribution'] for r in group})
        for radix_only, filename in [(False, 'throughput'), (True, 'radix_widths')]:
            fig, axes = plt.subplots(1, len(distributions), figsize=(5 * len(distributions), 4), squeeze=False)
            for ax, distribution in zip(axes[0], distributions):
                for algorithm in ALGORITHMS:
                    if radix_only and not algorithm.startswith('radix-'): continue
                    data = sorted((r for r in group if r['distribution'] == distribution and r['algorithm'] == algorithm), key=lambda r: int(r['key_count']))
                    if data: ax.plot([int(r['key_count']) for r in data], [float(r['keys_per_second']) / 1e6 for r in data], 'o-', label=algorithm)
                ax.set(xscale='log', xlabel='Keys', ylabel='Million keys / second', title=distribution)
                ax.grid(True, alpha=.3); ax.legend(fontsize=7)
            fig.suptitle(f'{label}\n{gpu}; seed={seed}; ' + ('Radix digit-width comparison' if radix_only else 'Sort throughput'))
            fig.tight_layout(); fig.savefig(output / f'{filename}_{index}.png', dpi=140); plt.close(fig)
        sizes = sorted({int(r['key_count']) for r in group})
        for size in sorted({sizes[0], sizes[-1]}):
            fig, axes = plt.subplots(1, len(distributions), figsize=(5 * len(distributions), 5), squeeze=False)
            for ax, distribution in zip(axes[0], distributions):
                data = [r for name in ALGORITHMS for r in group if r['algorithm'] == name and r['distribution'] == distribution and int(r['key_count']) == size]
                bottom = [0.] * len(data)
                for field, title in [('h2d','H2D'), ('sort','Sort'), ('d2h','D2H')]:
                    values = [float(r['median_' + field + '_ms']) for r in data]
                    ax.bar(range(len(data)), values, bottom=bottom, label=title)
                    bottom = [a+b for a,b in zip(bottom, values)]
                ax.set_xticks(range(len(data)), [r['algorithm'] for r in data], rotation=55, ha='right')
                ax.set(ylabel='Median milliseconds', title=distribution); ax.legend()
            fig.suptitle(f'{label}\nCUDA event breakdown (not wall time): n={size}; {gpu}; seed={seed}')
            fig.tight_layout(); fig.savefig(output / f'event_breakdown_{index}_{size}.png', dpi=140); plt.close(fig)
        workloads = defaultdict(list)
        for row in group: workloads[(row['distribution'], int(row['key_count']))].append(row)
        for (distribution, size), workload in sorted(workloads.items()):
            custom = [r for r in workload if r['algorithm'] != 'cub']
            if custom:
                best = max(custom, key=lambda r: float(r['keys_per_second']))
                percent = best['percentage_of_cub'] or 'unavailable'
                notes.append(f"{gpu}; seed={seed}; {distribution}; n={size}: fastest custom={best['algorithm']}, {best['keys_per_second']} keys/s, percentage of matching CUB={percent}")
            for row in workload:
                h,s,d = [float(row['median_' + f + '_ms']) for f in ['h2d','sort','d2h']]
                notes.append(f"  {row['algorithm']}: event transfer fraction={(h+d)/(h+s+d):.2%}; median wall={row['median_end_to_end_ms']} ms")
    (output / 'analysis.txt').write_text('\n'.join(notes) + '\n', encoding='utf-8')
    print(f'{label}: wrote charts and analysis to {output}')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--fixture', action='store_true')
    a = p.parse_args()
    generate(a.input, a.output, a.fixture)
