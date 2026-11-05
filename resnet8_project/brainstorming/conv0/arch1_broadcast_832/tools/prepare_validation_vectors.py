"""Prepare independent-image and arithmetic-stress fixtures, without running HDL.

Preserves the default fixture, generated C header and fixed quantization package.
The output must be a new directory. Each case can be passed to simulate.tcl.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from fixtures import load_int8_fixture

def oracle(image, weights, bias, multiplier, shift):
    padded=np.pad(image.reshape(32,32,3).astype(np.int64),((1,1),(1,1),(0,0)))
    windows=np.stack([padded[y:y+32,x:x+32,c] for y in range(3) for x in range(3) for c in range(3)],axis=-1)
    wide=windows @ weights.reshape(26,27).astype(np.int64).T + bias.astype(np.int64)
    raw=((wide+2**31)%2**32-2**31).astype(np.int32)
    # Python unbounded integer rounding is independent of fixed-width HDL expressions.
    output=[]
    for i,value in enumerate(raw.reshape(-1)):
        channel=i%26
        product=int(value)*int(multiplier[channel])
        rounded=(abs(product)+(1<<(int(shift[channel])-1)))>>int(shift[channel])
        output.append(max(0,min(127,-rounded if product<0 else rounded)))
    return raw.reshape(-1),np.array(output,dtype=np.int8)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=PROJECT/'arch1_broadcast_832/build/validation_vectors')
    p.add_argument('--golden',type=Path,default=PROJECT/'resnet8/artifacts/golden')
    a=p.parse_args()
    if a.out.exists(): raise SystemExit('Refusing to overwrite existing vector directory; choose a fresh --out.')
    f=load_int8_fixture()
    golden=json.loads((a.golden/'manifest.json').read_text())
    if golden['export_manifest_sha256']!=f['identity']['export_manifest_sha256']:
        raise ValueError('Golden/portable export identity mismatch')
    if golden['weights_sha256']!=f['identity']['checkpoint_sha256']:
        raise ValueError('Golden checkpoint identity mismatch')
    cases=[]
    for record in golden['records']:
        label=f'{record["test_index"]:05d}_{record["label"]}'
        folder=a.golden/label/'integer'
        meta=json.loads((folder/'tensors.json').read_text())
        arrays={}
        for name in ('input','conv0_acc','conv0'):
            path=folder/f'{name}.npy'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=meta[name]['sha256']:
                raise ValueError(f'Golden checksum mismatch: {path}')
            arrays[name]=np.load(path,allow_pickle=False).reshape(-1)
        raw,out=oracle(arrays['input'],f['weights'],f['bias'],f['multiplier'],f['shift'])
        np.testing.assert_array_equal(raw,arrays['conv0_acc'])
        np.testing.assert_array_equal(out,arrays['conv0'])
        cases.append((label,'saved independent CIFAR-10 image',arrays['input'],f['weights'],f['bias'],raw,out))
    rng=np.random.default_rng(260832)
    for name,image,weights,bias in [
        ('synthetic_zero',np.zeros(3072,dtype=np.int8),f['weights'],f['bias']),
        ('synthetic_signed',rng.integers(-128,128,3072,dtype=np.int16).astype(np.int8),f['weights'],f['bias']),
        ('synthetic_wrap_positive',np.full(3072,127,dtype=np.int8),np.full(702,127,dtype=np.int8),np.full(26,2**31-1,dtype=np.int32)),
        ('synthetic_wrap_negative',np.full(3072,-128,dtype=np.int8),np.full(702,127,dtype=np.int8),np.full(26,-2**31,dtype=np.int32)),
    ]:
        raw,out=oracle(image,weights,bias,f['multiplier'],f['shift'])
        cases.append((name,'synthetic arithmetic stress; not network accuracy',image,weights,bias,raw,out))
    # All source checks precede writing any output.
    summary=[]
    for name,kind,image,weights,bias,raw,out in cases:
        target=a.out/name
        manifest={'description':kind,'checkpoint_sha256':f['identity']['checkpoint_sha256'],
            'export_manifest_sha256':f['identity']['export_manifest_sha256'],
            'parameter_note':'Synthetic wrap cases replace weights/bias; quantization remains original. Export hash identifies the base only.', 'files':{}}
        values=[('layer_vectors/conv0/input.mem',image,8),('export/conv0_weights.mem',weights,8),
            ('export/conv0_bias.mem',bias,32),('export/conv0_quant.mem',f['quant'],32),
            ('layer_vectors/conv0/expected_acc.mem',raw,32),('layer_vectors/conv0/expected_output.mem',out,8)]
        for relative,array,bits in values:
            path=target/relative; path.parent.mkdir(parents=True,exist_ok=True)
            data=('\n'.join(f'{int(v)&((1<<bits)-1):0{bits//4}x}' for v in array.reshape(-1))+'\n').encode('ascii')
            path.write_bytes(data)
            manifest['files'][relative]={'elements':array.size,'bits_per_element':bits,'sha256':hashlib.sha256(data).hexdigest()}
        (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        load_int8_fixture(target) # Validate dimensions and every generated file hash.
        summary.append({'case':name,'kind':kind,'raw_values':26624,'int8_values':26624,
            'manifest_sha256':hashlib.sha256((target/'manifest.json').read_bytes()).hexdigest()})
    (a.out/'suite.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(f'PASS Python: {len(cases)} vector cases generated/hash-checked; {len(golden["records"])} archived images independently matched. No HDL run.')
    print(a.out)

if __name__=='__main__': main()
