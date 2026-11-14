"""Synthetic plotting/CSV validation tests. Never creates measured GPU results."""
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import plot_benchmarks as plot
from run_benchmarks import COMMON, RAW, SUMMARY

def fixture(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'metadata.json').write_text(json.dumps({'data_kind': 'synthetic_fixture',
        'warning': 'TEST FIXTURE ONLY - NO GPU MEASUREMENTS'}), encoding='utf-8')
    summary, raw = [], []
    for distribution in ['random', 'duplicate-heavy']:
        for n in [1024, 65536]:
            for index, name in enumerate(plot.ALGORITHMS):
                common = dict(algorithm=name, key_count=n, distribution=distribution, seed=12345,
                    warmups=3, trials=2, gpu='SYNTHETIC FIXTURE GPU', host_memory='pinned', build_configuration='Release')
                sort = (7 - index) * n / 65536
                for i, factor in enumerate([.8, 1.2]):
                    raw.append(dict(common, trial=i, h2d_ms=.1*factor, sort_ms=sort*factor,
                        d2h_ms=.2*factor, end_to_end_ms=(sort+.5)*factor, status='PASS', reason='SYNTHETIC FIXTURE'))
                summary.append(dict(common, completed_trials=2, median_h2d_ms=.1, median_sort_ms=sort,
                    median_d2h_ms=.2, median_end_to_end_ms=sort+.5, keys_per_second=n*1000/sort,
                    percentage_of_cub=100/(7-index), status='PASS', reason='SYNTHETIC FIXTURE'))
    failed = dict(summary[0], algorithm='ignored-failure', status='FAIL', reason='synthetic failure', completed_trials=0)
    summary.append(failed)
    for name, fields, rows in [('summary.csv',SUMMARY,summary), ('raw.csv',RAW,raw)]:
        with (directory/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)

class PipelineTests(unittest.TestCase):
    def test_smoke_gate_blocks_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'blocked'
            run=subprocess.run([sys.executable,str(ROOT/'scripts'/'run_benchmarks.py'),
                '--runner',str(Path(tmp)/'absent.exe'),'--output',str(path),
                '--sizes','1024','--distributions','random'],capture_output=True,text=True)
            self.assertEqual(run.returncode,1)
            with (path/'summary.csv').open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),7)
            self.assertTrue(all(r['status']=='SKIPPED' and not r['median_sort_ms'] for r in rows))
            self.assertEqual(json.loads((path/'metadata.json').read_text())['measured_cases'],0)
    def test_validation_and_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp); fixture(path)
            self.assertEqual(len(plot.load_validated(path)),28)
            with self.assertRaisesRegex(ValueError,'--fixture'):
                plot.generate(path,path/'forbidden')
            plot.generate(path,path/'charts',True)
            self.assertEqual(len(list((path/'charts').glob('*.png'))),4)
            self.assertIn('SYNTHETIC FIXTURE', (path/'charts'/'analysis.txt').read_text())
    def test_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp); fixture(path)
            with (path/'summary.csv').open(newline='',encoding='utf-8') as f:
                rows=list(csv.DictReader(f))
            rows[0]['percentage_of_cub']='999'
            with (path/'summary.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=SUMMARY);writer.writeheader();writer.writerows(rows)
            with self.assertRaisesRegex(ValueError,'percentage'):plot.load_validated(path)
    def test_missing_trials_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);fixture(path)
            with (path/'raw.csv').open(newline='',encoding='utf-8') as f:
                rows=list(csv.DictReader(f))[1:]
            with (path/'raw.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=RAW);writer.writeheader();writer.writerows(rows)
            with self.assertRaisesRegex(ValueError,'Missing raw'):plot.load_validated(path)

if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--write-fixture':
        fixture(Path(sys.argv[2]))
    else:unittest.main()
