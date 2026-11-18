"""Verify and restore the MUVA data-only backup into a new, empty directory."""
import argparse
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_name(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name:
        raise RuntimeError('Unsafe archive member: '+name)
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-dir',required=True,type=Path)
    parser.add_argument('--destination',required=True,type=Path)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    bundle=args.download_dir.resolve()
    out=args.destination.resolve()
    manifest=json.loads((bundle/'manifest.json').read_text())
    if manifest.get('format_version')!=1:
        raise RuntimeError('Unsupported manifest version')
    expected={'images.zip','annotations.tar.gz','fragmented-protocol.tar.gz','statistics.tar.gz'}
    if {r['file'] for r in manifest['files']}!=expected:
        raise RuntimeError('Unexpected backup file set')
    for record in manifest['files']:
        path=bundle/record['file']
        if path.stat().st_size!=record['bytes'] or sha256(path)!=record['sha256']:
            raise RuntimeError('Checksum mismatch: '+str(path))
    # Preflight every archive before creating any restored files.
    with zipfile.ZipFile(bundle/'images.zip') as z:
        for member in z.infolist():
            rel=safe_name(member.filename)
            if not rel.parts or rel.parts[0]!='images':
                raise RuntimeError('Unexpected ZIP prefix')
    for name in sorted(expected-{'images.zip'}):
        with tarfile.open(bundle/name,'r:gz') as tar:
            for member in tar:
                rel=safe_name(member.name)
                if not member.isfile() or rel.parts[0] not in {'MUVA','MUVA_fragmented_dataset','MUVA_experiments','MUVA_case_statistics'}:
                    raise RuntimeError('Unexpected TAR member: '+member.name)
    if args.verify_only:
        print('VERIFIED four archives and safe member paths; no data extracted')
        return
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Destination must be new or empty; refusing to overwrite '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    (out/'MUVA').mkdir()
    with zipfile.ZipFile(bundle/'images.zip') as z:
        z.extractall(out/'MUVA')
    for name in sorted(expected-{'images.zip'}):
        with tarfile.open(bundle/name,'r:gz') as tar:
            # All entries were preflighted as regular relative files, without links.
            for member in tar:
                dst=out/member.name
                dst.parent.mkdir(parents=True,exist_ok=True)
                source=tar.extractfile(member)
                if source is None:
                    raise RuntimeError('Unreadable archive member: '+member.name)
                with source, dst.open('xb') as target:
                    for block in iter(lambda:source.read(1024*1024),b''):
                        target.write(block)
    alias=out/'MUVA/images/random-limited'
    if not alias.exists():
        alias.symlink_to('random-limited-fixed')
    for split,count in manifest['splits'].items():
        rows=[json.loads(x) for x in (out/f'MUVA_experiments/data/manifests/{split}.jsonl').read_text().splitlines()]
        if len(rows)!=count:
            raise RuntimeError('Split count mismatch: '+split)
        for row in rows:
            paths=[row['inputs']['reference_rgb'],row['inputs']['target_rgb'],row['inputs']['reference_mask'],row['supervision']['target_visible_mask']]
            for path in paths:
                mapped=out/Path(path).relative_to('/root/datasets')
                if not mapped.is_file():
                    raise RuntimeError('Restored input missing: '+str(mapped))
    print('RESTORED',out,'splits',manifest['splits'],'all manifest inputs present')


if __name__=='__main__':
    main()
