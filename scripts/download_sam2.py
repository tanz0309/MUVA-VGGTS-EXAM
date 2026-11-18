"""Resumable ranged download, verified against the published HuggingFace LFS SHA256."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import time
import requests

root=Path(__file__).parents[1]/'weights/sam2'
size=897952466
expected='7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b'
url='https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt'
chunk=32*1024*1024
parts=root/'download_parts';parts.mkdir(exist_ok=True)
initial=root/'sam2_hiera_large.pt.part'
if initial.exists():
    with initial.open('rb') as f:
        for i in range(initial.stat().st_size//chunk):
            block=f.read(chunk);p=parts/f'{i:03d}.part'
            if not p.exists(): p.write_bytes(block)

def fetch(i):
    start=i*chunk;end=min(size,start+chunk)-1;p=parts/f'{i:03d}.part'
    if p.exists() and p.stat().st_size==end-start+1:return p
    for retry in range(12):
        try:
            endpoint=url if retry%2==0 else 'https://huggingface.co/jaychempan/sam2/resolve/main/sam2_hiera_large.pt'
            with requests.get(endpoint+f'?muva_part={i}&retry={retry}',headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,60)) as r:
                r.raise_for_status()
                assert r.status_code==206 and r.headers['Content-Range']==f'bytes {start}-{end}/{size}',dict(r.headers)
                tmp=p.with_suffix('.tmp')
                with tmp.open('wb') as f:
                    for b in r.iter_content(1024*1024):f.write(b)
                assert tmp.stat().st_size==end-start+1
                tmp.replace(p)
            print('downloaded_chunk',i,flush=True);return p
        except Exception as exc:
            print('retry',i,retry,type(exc).__name__,str(exc)[:120],flush=True)
            if retry==11:raise
            time.sleep(2)

with ThreadPoolExecutor(max_workers=4) as pool:
    completed=list(pool.map(fetch,range((size+chunk-1)//chunk)))
h=hashlib.sha256()
with initial.open('wb') as f:
    for p in completed:
        b=p.read_bytes();h.update(b);f.write(b)
assert h.hexdigest()==expected,(h.hexdigest(),expected)
initial.replace(root/'sam2_hiera_large.pt')
print('VERIFIED_SHA256',expected,flush=True)
for p in completed:p.unlink()
parts.rmdir()
