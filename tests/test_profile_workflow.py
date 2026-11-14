import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from profile_kernels import capture_plan,classify,parse_metrics

class ProfilingTests(unittest.TestCase):
    def test_launch_mapping(self):
        # Independent enumeration of the actual Phase 4/5 launch schedules.
        for n in [65536,262144]:
            global_steps=[];cross=[];tiles=[('initial',256)]
            stage=2
            while stage<=n:
                stride=stage//2
                while stride:
                    global_steps.append((stage,stride))
                    if stage>256 and stride>=256:cross.append((stage,stride))
                    stride//=2
                if stage>256:tiles.append(('finish',stage))
                stage*=2
            plan=capture_plan(n)
            self.assertEqual(global_steps[plan[0]['skip']],(n,n//2))
            self.assertEqual(global_steps[plan[1]['skip']],(256,128))
            self.assertEqual(cross[plan[2]['skip']],(n,n//2))
            self.assertEqual(tiles[plan[3]['skip']],('initial',256))
            self.assertEqual(tiles[plan[4]['skip']],('finish',n))
            self.assertEqual(len(plan),11)
            self.assertEqual(len(capture_plan(n,duplicate=True)),17)
    def test_errors_and_parser(self):
        self.assertEqual(classify(dict(returncode=1,stdout='',stderr='ERR_NVGPUCTRPERM')), 'COUNTER_PERMISSION_DENIED')
        self.assertEqual(classify(dict(returncode=None,stdout='',stderr='missing')), 'MISSING_EXECUTABLE')
        text='preamble\n"ID","Metric Name","Metric Unit","Metric Value"\n"0","dram__bytes_read.sum","byte","1,024"\n'
        self.assertEqual(parse_metrics(text),[('dram__bytes_read.sum','1024','byte')])
        self.assertEqual(parse_metrics('No report'),[])
    def test_missing_profiler_has_no_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'reports'
            p=subprocess.run([sys.executable,str(ROOT/'scripts'/'profile_kernels.py'),'--runner',str(Path(tmp)/'missing_runner'),
                '--ncu',str(Path(tmp)/'missing_ncu'),'--output',str(output)],capture_output=True,text=True)
            self.assertEqual(p.returncode,1)
            meta=json.loads((output/'metadata.json').read_text())
            self.assertEqual(meta['blocker'],'MISSING_EXECUTABLE')
            self.assertEqual(meta['measured_metric_rows'],0)
            with (output/'summary.csv').open(newline='') as f:rows=list(csv.DictReader(f))
            self.assertTrue(all(r['status']=='UNAVAILABLE' and r['value']=='' for r in rows))

if __name__=='__main__':unittest.main()
