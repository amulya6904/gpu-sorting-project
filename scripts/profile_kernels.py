"""Small discovery-first Nsight Compute workflow. Never infers hardware findings."""
import argparse
import csv
import io
import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path

METRICS = {
    'dram__throughput.avg.pct_of_peak_sustained_elapsed': ('DRAM bandwidth', 'direct throughput relative to sustained peak; not standalone proof of a memory bottleneck'),
    'dram__bytes_read.sum': ('DRAM read traffic', 'direct device-memory bytes; caching can reduce traffic relative to source loads'),
    'dram__bytes_write.sum': ('DRAM write traffic', 'direct device-memory bytes'),
    'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum': ('global load sectors', 'direct L1/TEX sector count; traffic proxy, not cache-miss or efficiency percentage'),
    'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum': ('global load requests', 'direct request count; sectors/request can describe transaction footprint, not universal efficiency'),
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum': ('shared load bank conflicts', 'direct conflict counter where supported; distinguish from shared atomic contention'),
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum': ('shared store bank conflicts', 'direct conflict counter where supported; count must be interpreted with shared access volume'),
    'smsp__thread_inst_executed_per_inst_executed.ratio': ('active-thread proxy', 'average thread instructions per warp instruction; low values can reflect predication, bounds, pair ownership, or loop tails, not just branch divergence'),
    'smsp__sass_average_branch_targets_threads_uniform.pct': ('branch-uniformity proxy', 'uniform branch targets where measured; does not count all predication or explain its cause'),
}
FIELDS = ['algorithm','kernel','key_count','distribution','seed','capture','metric','value','unit','status','interpretation','report']

def capture_plan(n, widths=(1,8), duplicate=False):
    if n not in (65536,262144): raise ValueError('Representative sizes: 65536 or 262144')
    if len(set(widths)) != 2 or any(w not in (1,2,4,8) for w in widths):
        raise ValueError('Choose exactly two distinct supported radix widths')
    levels = n.bit_length()-1
    cross_skip = sum(range(1,levels-8))
    plan = [
        dict(algorithm='bitonic-global',kernel='bitonic_step',skip=levels*(levels-1)//2,
             capture=f'warmup0 stage={n} stride={n//2} cross-tile'),
        dict(algorithm='bitonic-global',kernel='bitonic_step',skip=28,
             capture='warmup0 stage=256 stride=128 within-tile global-memory step'),
        dict(algorithm='bitonic-shared',kernel='bitonic_step',skip=cross_skip,
             capture=f'warmup0 stage={n} stride={n//2} cross-tile'),
        dict(algorithm='bitonic-shared',kernel='tile_steps',skip=0,
             capture='warmup0 initial=true stages=2..256; alternating initial tile runs'),
        dict(algorithm='bitonic-shared',kernel='tile_steps',skip=levels-8,
             capture=f'warmup0 initial=false stage={n} strides=128..1'),
    ]
    for width in widths:
        for kernel in ['histogram_step','prefix_step','scatter_step']:
            plan.append(dict(algorithm=f'radix-{width}',kernel=kernel,skip=0,
                             capture=f'warmup0 pass=0 shift=0 bits={width}'))
    for entry in plan: entry['distribution']='random'
    if duplicate:
        plan += [dict(entry,distribution='duplicate-heavy') for entry in plan if entry['algorithm'].startswith('radix-')]
    return plan

def capture(command):
    try:
        p=subprocess.run(command,text=True,capture_output=True)
        return dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    except OSError as error:
        return dict(command=command,returncode=None,stdout='',stderr=str(error))

def classify(result):
    text=result['stdout']+'\n'+result['stderr']
    if 'ERR_NVGPUCTRPERM' in text: return 'COUNTER_PERMISSION_DENIED'
    if result['returncode'] is None: return 'MISSING_EXECUTABLE'
    if re.search(r'no (CUDA|supported|profilable).*device|no CUDA GPU|no devices',text,re.I): return 'NO_SUPPORTED_GPU'
    if result['returncode'] != 0: return 'TOOL_OR_RUNTIME_ERROR'
    return 'OK'

def parse_metrics(text):
    """Parse ncu --import --page raw --csv, ignoring preamble and unit rows."""
    lines=text.splitlines()
    start=next((i for i,l in enumerate(lines) if '"Metric Name"' in l and '"Metric Value"' in l),None)
    if start is None: return []
    rows=[]
    for row in csv.DictReader(io.StringIO('\n'.join(lines[start:]))):
        name=row.get('Metric Name','')
        if name not in METRICS: continue
        value=row.get('Metric Value','').replace(',','').strip()
        try: number=float(value)
        except ValueError: continue
        if math.isfinite(number): rows.append((name,value,row.get('Metric Unit','')))
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runner',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path,help='New report directory')
    p.add_argument('--ncu',default='ncu',help='CLI executable or absolute path')
    p.add_argument('--size',type=int,choices=[65536,262144],default=65536)
    p.add_argument('--seed',type=int,default=12345)
    p.add_argument('--radix-widths',nargs=2,type=int,default=[1,8])
    p.add_argument('--duplicate-heavy',action='store_true')
    a=p.parse_args()
    if not 0<=a.seed<=4294967295:p.error('Seed must fit uint32')
    try:plan=capture_plan(a.size,tuple(a.radix_widths),a.duplicate_heavy)
    except ValueError as error:p.error(str(error))
    a.output.mkdir(parents=True,exist_ok=False)
    metadata=dict(state='discovery',os=platform.platform(),commands=[],plan=plan,
                  key_count=a.size,seed=a.seed,profiled_times_are_not_benchmark_times=True,
                  orchestration_command=[sys.executable,*sys.argv],radix_selection='explicit widths; default extremes if Phase 8 measurements unavailable')
    def save(): (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    def run(command,name):
        result=capture(command);metadata['commands'].append(result)
        (a.output/(name+'.log')).write_text(result['stdout']+'\n'+result['stderr'],encoding='utf-8');save()
        return result
    run(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],'gpu')
    run(['nvcc','--version'],'toolkit')
    version=run([a.ncu,'--version'],'profiler_version')
    help_result=run([a.ncu,'--help'],'profiler_help')
    sections=run([a.ncu,'--list-sections'],'sections')
    metrics=run([a.ncu,'--query-metrics','--query-metrics-mode','all','--devices','0'],'supported_metrics')
    available=metrics['stdout']+'\n'+metrics['stderr']
    if '--query-metrics-collection' in help_result['stdout']:
        source=run([a.ncu,'--query-metrics','--query-metrics-mode','all','--devices','0',
                    '--query-metrics-collection','source'],'supported_source_metrics')
        if classify(source)=='OK':available+='\n'+source['stdout']
    selected=[m for m in METRICS if re.search(r'(?<![A-Za-z0-9_.])'+re.escape(m)+r'(?![A-Za-z0-9_.])',available)]
    metadata['selected_metrics']=selected
    metadata['unavailable_metrics']=[m for m in METRICS if m not in selected]
    rows=[]
    def add(entry,metric,value,unit,status,interpretation,report=''):
        rows.append(dict(algorithm=entry['algorithm'],kernel=entry['kernel'],key_count=a.size,
            distribution=entry['distribution'],seed=a.seed,capture=entry['capture'],metric=metric,
            value=value,unit=unit,status=status,interpretation=interpretation,report=report))
    discovery=next((classify(r) for r in [version,sections,metrics] if classify(r)!='OK'),None)
    if discovery or not selected:
        metadata['state']='blocked_discovery'
        metadata['blocker']=discovery or 'NO_REQUESTED_METRICS_AVAILABLE'
        for entry in plan:
            for metric in METRICS:add(entry,metric,'','','UNAVAILABLE',metadata['blocker']+'; no hardware value collected. '+METRICS[metric][1])
    else:
        runner=str(a.runner.resolve())
        # Normal unprofiled smoke first distinguishes runtime failure from counter access.
        smoke=run([runner,'--algorithm','all','--size','1024','--distribution','random','--seed',str(a.seed),'--warmups','1','--trials','1'],'unprofiled_smoke')
        if classify(smoke)!='OK' or 'Summary: passed=7 ' not in smoke['stdout']:
            metadata['state']='blocked_runtime_smoke';metadata['blocker']=classify(smoke)
            for entry in plan:
                for metric in METRICS:add(entry,metric,'','','UNAVAILABLE','Unprofiled correctness smoke failed; inspect unprofiled_smoke.log')
        else:
            blocked=None
            for i,entry in enumerate(plan):
                stem=f"{i:02d}_{entry['algorithm']}_{entry['kernel']}_{entry['distribution']}"
                report=a.output/(stem+'.ncu-rep')
                if blocked:
                    for metric in METRICS:add(entry,metric,'','','UNAVAILABLE','Earlier capture blocked: '+blocked)
                    continue
                command=[a.ncu,'--kernel-name-base','demangled','--kernel-name','regex:.*'+entry['kernel']+'.*',
                    '--launch-skip',str(entry['skip']),'--launch-count','1','--metrics',','.join(selected),
                    '--replay-mode','kernel','--cache-control','all','--clock-control','none','--export',str(report),runner,
                    '--algorithm',entry['algorithm'],'--size',str(a.size),'--distribution',entry['distribution'],
                    '--seed',str(a.seed),'--warmups','1','--trials','1']
                result=run(command,stem)
                state=classify(result)
                if state!='OK' or not report.exists():
                    blocked=state if state!='OK' else 'NO_REPORT_EXPORTED'
                    for metric in METRICS:add(entry,metric,'','','UNAVAILABLE',blocked+'; inspect capture log')
                    continue
                exported=run([a.ncu,'--import',str(report),'--page','raw','--csv'],stem+'_raw')
                values={m:(v,u) for m,v,u in parse_metrics(exported['stdout'])}
                if classify(exported)!='OK' or not values:
                    blocked='REPORT_EXPORT_OR_PARSE_FAILED'
                for metric in METRICS:
                    if metric in values:
                        value,unit=values[metric];add(entry,metric,value,unit,'MEASURED',METRICS[metric][1],str(report))
                    else:add(entry,metric,'','','UNAVAILABLE','Not supported or not reported by this profiler/GPU',str(report))
            metadata['state']='captures_collected' if not blocked else 'blocked_capture'
            metadata['blocker']=blocked
    with (a.output/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
    metadata['measured_metric_rows']=sum(r['status']=='MEASURED' for r in rows)
    save()
    print(f"{metadata['state']}: {metadata['measured_metric_rows']} measured metric rows; {a.output}")
    return 0 if metadata['state']=='captures_collected' and metadata['measured_metric_rows'] else 1

if __name__=='__main__':raise SystemExit(main())
