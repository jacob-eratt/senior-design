"""python tools/summarize_uart.py UART_CAPTURE.txt [--clock-mhz 100]"""
import argparse
import csv
import math
import statistics
import re

def parse_capture(text, cold_frames=20, warm_frames=200):
    """Accept one complete successful run; reject partial or mixed captures."""
    if cold_frames < 1 or warm_frames < 1:
        raise ValueError('Expected frame counts must be positive')
    rows=[]
    passed=False
    banners=0
    for line in text.splitlines():
        line=line.strip()
        if re.search(r'\bFAIL\b|\bmismatch\b', line, re.I):
            raise ValueError('Capture contains failure/mismatch')
        if line.startswith('Conv0 KV260 benchmark:'):
            banners+=1
            if banners>1 or rows or passed: raise ValueError('Multiple/reordered runs')
        if line.startswith('PASS'):
            match=re.fullmatch(r'PASS: (\d+) frames, (\d+) INT8 comparisons per frame',line)
            if not match or passed or tuple(map(int,match.groups()))!=(cold_frames+warm_frames,26624):
                raise ValueError('Invalid or duplicate PASS totals')
            if len(rows)!=cold_frames+warm_frames: raise ValueError('Incomplete FRAME records')
            passed=True
        if not line.startswith('FRAME,'): continue
        if passed: raise ValueError('FRAME after PASS')
        row=next(csv.reader([line]))
        if len(row)!=10 or row[1] not in ('cold','warm'): raise ValueError('Malformed FRAME')
        nums=[int(v) for v in row[2:]]
        if any(v<0 for v in nums) or nums[1]==0: raise ValueError('Invalid measurement')
        index,total,run,raw,first,istall,ostall,loads=nums
        mode='cold' if len(rows)<cold_frames else 'warm'
        wanted=len(rows) if mode=='cold' else len(rows)-cold_frames
        if row[1]!=mode or index!=wanted or wanted >= (cold_frames if mode=='cold' else warm_frames):
            raise ValueError('Missing, duplicate, reordered, or excess frame')
        if not (0<first<=raw<=run) or ostall>run: raise ValueError('Inconsistent hardware counters')
        # ceil(3072/16)+ceil(702/16)+ceil(104/16)+ceil(208/16) = 256.
        if loads!=(256 if mode=='cold' else 192): raise ValueError('Unexpected loader count')
        rows.append(row)
    if not passed: raise ValueError('Missing final PASS; incomplete capture')
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture')
    p.add_argument('--clock-mhz', type=float, default=100)
    p.add_argument('--cold-frames', type=int, default=20)
    p.add_argument('--warm-frames', type=int, default=200)
    a=p.parse_args()
    try:
        if not math.isfinite(a.clock_mhz) or a.clock_mhz<=0: raise ValueError('Invalid clock')
        with open(a.capture, encoding='utf-8') as f:
            rows=parse_capture(f.read(),a.cold_frames,a.warm_frames)
        if any(int(r[3])/1000 < int(r[4])/a.clock_mhz for r in rows):
            raise ValueError('Total time shorter than hardware time; check clocks/timer')
    except (ValueError,OSError) as exc:
        p.exit(2,f'REJECTED: {exc}\n')
    for mode in ('cold','warm'):
        selected=[r for r in rows if r[1]==mode]
        if not selected: continue
        values=sorted(int(r[3])/1000 for r in selected)
        print(f'{mode}: n={len(values)}, end-to-end us: min={values[0]:.3f}, '
              f'mean={statistics.mean(values):.3f}, median={statistics.median(values):.3f}, '
              f'p95={values[math.ceil(.95*len(values))-1]:.3f}, max={values[-1]:.3f}')
        print(f'  mean START-to-last-output handshake: {statistics.mean(int(r[4]) for r in selected)/a.clock_mhz:.3f} us')
        print(f'  mean output stalls: {statistics.mean(int(r[8]) for r in selected):.1f} cycles')
    print('Validated complete PASS capture; reported counters are not independent board certification.')

if __name__=='__main__': main()
