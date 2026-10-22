"""Python-only checks of RTL address/bank equations and generated fixture bytes.

This does not compile or execute RTL. Run after export_fixture.py.
"""
from collections import deque
from pathlib import Path
import hashlib
import json
import random
import re
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from cycle_sim.fixtures import load_int8_fixture

def main():
    f=load_int8_fixture()
    image=f['image'].reshape(192,4,4)  # 16-byte word, bank, byte
    weights=f['weights'].reshape(26,27).astype(np.int64)
    results=[]
    for row in range(32):
        accum=np.broadcast_to(f['bias'],(32,26)).astype(np.int64).copy()
        for tap in range(27):
            for pos in range(32):
                sy=row+tap//9-1
                sx=pos+(tap//3)%3-1
                if 0<=sy<32 and 0<=sx<32:
                    addr=(sy*32+sx)*3+tap%3
                    sample=int(image[addr>>4,(addr%16)//4,addr%4])
                else: sample=0
                accum[pos]+=sample*weights[:,tap]
        results.append(accum.astype(np.int32))
    raw=np.array(results).reshape(-1)
    np.testing.assert_array_equal(raw,f['expected'])
    product=raw.reshape(-1,26).astype(np.int64)*f['multiplier']
    magnitude=np.abs(product)
    rounded=(magnitude+(np.int64(1)<<(f['shift'].astype(np.int64)-1)))>>f['shift']
    output=np.clip(np.where(product<0,-rounded,rounded),0,127).astype(np.int8).reshape(-1)
    np.testing.assert_array_equal(output,f['expected_int8'])

    # Exercise the 16-bank ring equations through repeated wrap and full/empty.
    rng=random.Random(832)
    mem=[[None]*16 for _ in range(16)]
    queue=deque()
    wr=rd=count=seq=0
    pushes=pops=0
    for _ in range(100000):
        push=count<=243 and rng.random()<.6
        pop=count>=4 and rng.random()<.8
        # Read old contents first, matching edge handshakes.
        if pop:
            for lane in range(4):
                bank=(rd+lane)%16
                delta=(bank-rd%16)%16
                ra=(rd+delta)%256
                assert mem[bank][ra>>4]==queue.popleft()
            pops+=1
        if push:
            for bank in range(16):
                delta=(bank-wr%16)%16
                if delta<13:
                    wa=(wr+delta)%256
                    mem[bank][wa>>4]=seq+delta
            queue.extend(range(seq,seq+13)); seq+=13; pushes+=1
        if push: wr=(wr+13)%256
        if pop: rd=(rd+4)%256
        count+=13*push-4*pop
        assert count==len(queue) and 0<=count<=256
    target=Path(__file__).resolve().parents[1]
    header=(target/'cpu/fixture_data.h').read_text()
    manifest=json.loads((target/'cpu/fixture_manifest.json').read_text())
    for name, meta in manifest['payloads'].items():
        body=re.search(r'\b'+name+r'\[\d+\].*?=\s*\{(.*?)\};',header,re.S).group(1)
        data=bytes(int(x,16) for x in re.findall(r'0x([0-9a-f]{2})',body))
        assert len(data)==meta['bytes'] and hashlib.sha256(data).hexdigest()==meta['sha256']
    print(f'PASS Python: 26624 raw + 26624 INT8 fixture values; ring {pushes} pushes/{pops} pops; C payload hashes.')
    print('Not an HDL simulation, synthesis, or AXI protocol certification.')

if __name__=='__main__': main()
