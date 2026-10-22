"""Archive completed Vivado evidence and exact source hashes; never runs tools.

Reports are copied into a fresh directory. Incomplete HDL suites are rejected.
Presence of reports does not imply routed timing closure or successful board execution.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=PROJECT/'reports'/('vivado_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    a=p.parse_args()
    if a.out.exists(): raise SystemExit('Choose a fresh archive directory.')
    suites=sorted(d for d in (ROOT/'build').glob('validation_*') if (d/'baseline.log').exists())
    if not suites: raise SystemExit('No validation suite found.')
    suite=suites[-1]
    logs=sorted(suite.glob('*.log'))
    if len(logs)!=15: raise SystemExit('Expected baseline plus 14 completed logs.')
    frames=[]
    for log in logs:
        text=log.read_text(errors='strict')
        if 'PASS tb_conv0_axi:' not in text or re.search(r'Fatal:|ERROR:',text,re.I):
            raise SystemExit(f'Unsuccessful HDL log: {log}')
        matches=re.findall(r'PASS frame (\d+) warm=(\d) stalls=(\d) bytes=(\d+) run_cycles=(\d+)',text)
        if len(matches)!=2: raise SystemExit(f'Incomplete frame results: {log}')
        for n,w,s,b,c in matches:
            if int(b)!=26624: raise SystemExit('Unexpected output count')
            frames.append(dict(case=log.stem,frame=int(n),warm=bool(int(w)),stalls=bool(int(s)),bytes=int(b),run_cycles=int(c)))
    summary={'tool':'Vivado/XSim 2024.2','hdl_frames':len(frames),'distinct_saved_images':10,
        'synthetic_cases':4,'baseline_repeats_one_saved_image':True,
        'raw_comparisons':len(frames)*26624,'int8_comparisons':len(frames)*26624,
        'frames':frames,'components':{},'board_execution':False,'cpu_benchmark_execution':False,
        'source_sha256':{},'archive_sha256':{},
        'provenance_note':'Source hashes are archive-time fingerprints, not proof of original build revision. The 30-frame suite used the direct top; the later wrapper baseline is separate.',
        'vitis_compiled':False}
    a.out.mkdir(parents=True)
    def copy(path,relative):
        target=a.out/relative; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        summary['archive_sha256'][str(relative).replace('\\','/')]=digest(target)
    for log in logs: copy(log,Path('simulation')/log.name)
    for component in ('mac_int8_mixed','conv0_requant','conv0_result_fifo','conv0_832_core'):
        folder=ROOT/'build/components'/component
        util=folder/'utilization_synth.rpt'
        if not util.exists(): continue
        line=next(line for line in util.read_text().splitlines() if '(top)' in line)
        cols=[s.strip() for s in line.split('|')[1:-1]]
        summary['components'][component]=dict(zip(('LUT','logic_LUT','LUTRAM','SRL','FF','RAMB36','RAMB18','URAM','DSP'),map(int,cols[2:])))
        for report in folder.glob('*.rpt'): copy(report,Path('components')/component/report.name)
    pointer=ROOT/'build/last_integrated_project.txt'
    if pointer.exists():
        build=Path(pointer.read_text().strip()).parent
        summary['integrated_project']=str(build/'conv0_kv260.xpr')
        for report in (build/'reports').glob('*.rpt'): copy(report,Path('integrated')/report.name)
        summary['integrated_artifacts']={name:{'exists':(build/name).exists(),
            'sha256':digest(build/name) if (build/name).exists() else None}
            for name in ('conv0_kv260.bit','conv0_kv260.xsa')}
    for pattern in ('src/*.sv','src/*.v','sim/*.sv','scripts/*.tcl','cpu/benchmark.c','cpu/fixture_manifest.json'):
        for path in ROOT.glob(pattern): summary['source_sha256'][str(path.relative_to(ROOT)).replace('\\','/')]=digest(path)
    for name in ('validation_synth.log','integrated.log','sim_initial.log','export.log'):
        path=ROOT/'build'/name
        if path.exists(): copy(path,Path('tool_logs')/name)
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k in ('hdl_frames','raw_comparisons','int8_comparisons','components','integrated_artifacts')},indent=2))
    print(a.out)

if __name__=='__main__': main()
