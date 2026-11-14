"""Smoke-gated Phase 8 matrix. No synthetic measurements are produced here."""
import argparse
import csv
import datetime
import json
import platform
import shlex
import subprocess
import sys
import shutil
import os
from pathlib import Path

ALGORITHMS = ['bitonic-global', 'bitonic-shared', 'radix-1', 'radix-2', 'radix-4', 'radix-8', 'cub']
COMMON = ['algorithm','key_count','distribution','seed','warmups','trials','gpu','host_memory','build_configuration']
SUMMARY = COMMON + ['completed_trials','median_h2d_ms','median_sort_ms','median_d2h_ms','median_end_to_end_ms','keys_per_second','percentage_of_cub','status','reason']
RAW = COMMON + ['trial','h2d_ms','sort_ms','d2h_ms','end_to_end_ms','status','reason']

def capture(command):
    try:
        p = subprocess.run(command, text=True, capture_output=True)
        return {'command': command, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except OSError as error:
        return {'command': command, 'returncode': None, 'stdout': '', 'stderr': str(error)}

def combine(paths, destination):
    fields = None
    rows = []
    for path in paths:
        with path.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if fields is None:
                fields = reader.fieldnames
            if reader.fieldnames != fields:
                raise ValueError('CSV schema differs: ' + str(path))
            rows.extend(reader)
    if fields:
        with destination.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runner', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path, help='New directory; existing runs are never overwritten')
    p.add_argument('--sizes', nargs='+', type=int, default=[1024, 65536, 262144, 1048576])
    p.add_argument('--distributions', nargs='+', default=['random', 'sorted', 'reverse-sorted', 'duplicate-heavy'])
    p.add_argument('--seeds', nargs='+', type=int, default=[12345])
    p.add_argument('--algorithms', nargs='+', choices=ALGORITHMS, default=ALGORITHMS)
    p.add_argument('--warmups', type=int, default=3)
    p.add_argument('--trials', type=int, default=10)
    a = p.parse_args()
    if a.warmups < 1 or a.trials < 1 or any(n < 0 or n > 2147483647 for n in a.sizes):
        p.error('Positive counts and sizes in 0..INT_MAX required')
    if any(s < 0 or s > 4294967295 for s in a.seeds):
        p.error('Seeds must fit uint32')
    valid = ['random', 'sorted', 'reverse-sorted', 'duplicate-heavy', 'all-equal', 'boundary-value']
    if any(d not in valid for d in a.distributions):
        p.error('Unknown distribution')
    for values in [a.sizes, a.distributions, a.seeds, a.algorithms]:
        if len(values) != len(set(values)):
            p.error('Duplicate matrix entries are not allowed')
    a.output.mkdir(parents=True, exist_ok=False)
    cases = a.output / 'cases'
    cases.mkdir()
    runner = str(a.runner.resolve())
    metadata = {'data_kind': 'measured', 'state': 'starting', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'os': platform.platform(), 'python': sys.version, 'host_memory': 'pinned',
                'build_configuration_required': 'Release', 'matrix': vars(a).copy(),
                'orchestration_command': [sys.executable, *sys.argv], 'commands': []}
    metadata['matrix'] = {k: str(v) if isinstance(v, Path) else v for k, v in metadata['matrix'].items()}
    smi = shutil.which('nvidia-smi')
    if not smi and Path('C:/Windows/System32/nvidia-smi.exe').exists(): smi = 'C:/Windows/System32/nvidia-smi.exe'
    nvcc = shutil.which('nvcc')
    cuda_nvcc = Path(os.environ.get('CUDA_PATH', 'CUDA_NOT_INSTALLED')) / 'bin' / 'nvcc.exe'
    if not nvcc and cuda_nvcc.exists(): nvcc = str(cuda_nvcc)
    metadata['nvidia_smi'] = capture([smi or 'nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'])
    metadata['nvcc'] = capture([nvcc or 'nvcc', '--version'])
    meta_path = a.output / 'metadata.json'
    def save():
        meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    def run(command, log):
        result = capture(command)
        metadata['commands'].append(result)
        log.write_text(result['stdout'] + '\n' + result['stderr'], encoding='utf-8')
        save()
        return result
    save()
    smoke = run([runner, '--algorithm', 'all', '--size', '1024', '--distribution', 'random', '--seed', '12345',
                 '--warmups', '1', '--trials', '1'], a.output / 'smoke.log')
    if smoke['returncode'] != 0 or any('PASS: ' + name + ' median_sort_ms=' not in smoke['stdout'] for name in ALGORITHMS):
        metadata['state'] = 'blocked_before_matrix'
        metadata['reason'] = 'All-seven correctness smoke did not pass; no matrix measurements attempted'
        metadata['measured_cases'] = 0
        metadata['blocked_cases'] = len(a.sizes) * len(a.distributions) * len(a.seeds) * len(a.algorithms)
        with (a.output / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY); writer.writeheader()
            for seed in a.seeds:
                for distribution in a.distributions:
                    for size in a.sizes:
                        for name in a.algorithms:
                            writer.writerow(dict(algorithm=name, key_count=size, distribution=distribution, seed=seed,
                                warmups=a.warmups, trials=a.trials, gpu='unavailable', host_memory='pinned',
                                build_configuration='unverified', completed_trials=0, status='SKIPPED', reason=metadata['reason']))
        with (a.output / 'raw.csv').open('w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=RAW).writeheader()
        save()
        print(metadata['reason'] + '\nSee ' + str(a.output / 'smoke.log'))
        return 1
    raw_paths, summary_paths = [], []
    all_rows = []
    for seed in a.seeds:
        for distribution in a.distributions:
            for size in a.sizes:
                prefix = cases / f'{distribution}_{size}_{seed}'
                command = [runner, '--algorithm', ','.join(a.algorithms), '--size', str(size),
                           '--distribution', distribution, '--seed', str(seed), '--warmups', str(a.warmups),
                           '--trials', str(a.trials), '--output-prefix', str(prefix)]
                result = run(command, prefix.with_suffix('.log'))
                raw_path = Path(str(prefix) + '_raw.csv')
                summary_path = Path(str(prefix) + '_summary.csv')
                if not raw_path.exists() or not summary_path.exists():
                    # Setup/context failures occur before runner export. Record no times.
                    reason = (result['stderr'] or result['stdout'] or 'Runner exported no CSV').strip()
                    metadata['state'] = 'failed_during_matrix'
                    metadata['reason'] = reason
                    save()
                    # Preserve every requested algorithm as a failed case with blank metrics.
                    with summary_path.open('w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=SUMMARY); writer.writeheader()
                        for name in a.algorithms:
                            writer.writerow(dict(algorithm=name, key_count=size, distribution=distribution, seed=seed,
                                warmups=a.warmups, trials=a.trials, gpu='unknown', host_memory='pinned', build_configuration='unknown',
                                completed_trials=0, status='FAIL', reason=reason))
                else:
                    raw_paths.append(raw_path)
                    if result['returncode'] != 0:
                        with summary_path.open(newline='', encoding='utf-8') as f:
                            existing = list(csv.DictReader(f))
                        # A post-export cleanup/process failure invalidates apparent passes.
                        if all(r['status'] == 'PASS' for r in existing):
                            for row in existing:
                                row['status'] = 'FAIL'
                                row['reason'] = 'Runner exited nonzero after export: ' + result['stderr'].strip()
                                for field in SUMMARY[10:16]: row[field] = ''
                            with summary_path.open('w', newline='', encoding='utf-8') as f:
                                writer = csv.DictWriter(f, fieldnames=SUMMARY); writer.writeheader(); writer.writerows(existing)
                summary_paths.append(summary_path)
                with summary_path.open(newline='', encoding='utf-8') as f:
                    all_rows.extend(csv.DictReader(f))
                if not raw_paths:
                    with (a.output / 'raw.csv').open('w', newline='', encoding='utf-8') as f:
                        csv.DictWriter(f, fieldnames=RAW).writeheader()
                combine(raw_paths, a.output / 'raw.csv')
                combine(summary_paths, a.output / 'summary.csv')
    metadata['passed_cases'] = sum(r['status'] == 'PASS' for r in all_rows)
    metadata['failed_or_skipped_cases'] = sum(r['status'] != 'PASS' for r in all_rows)
    metadata['state'] = 'complete' if all(r['status'] == 'PASS' for r in all_rows) else 'complete_with_failures_or_skips'
    save()
    print(f"{metadata['state']}: {metadata['passed_cases']} passed; {metadata['failed_or_skipped_cases']} failed/skipped. CSVs: {a.output}")
    return 0 if metadata['state'] == 'complete' else 1

if __name__ == '__main__':
    raise SystemExit(main())
