"""Check backed-up source hashes, forbidden artifacts and common credential patterns."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {'.pt', '.pth', '.ckpt', '.safetensors', '.onnx', '.bin', '.pkl',
                      '.pickle', '.npy', '.npz', '.whl', '.so', '.zip', '.gz', '.ipynb',
                      '.png', '.jpg', '.jpeg', '.gif', '.mp4', '.pdf', '.pem', '.key'}
SECRETS = [
    re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}'),
    re.compile(rb'github_pat_[A-Za-z0-9_]{30,}'),
    re.compile(rb'hf_[A-Za-z0-9]{25,}'),
    re.compile(rb'-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----'),
    re.compile(rb'https?://[^\s/@:]+:[^\s/@]+@'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare-live', action='store_true', help='Also compare to original server source files')
    parser.add_argument('--staged', action='store_true', help='Require every snapshotted source to be tracked/staged')
    args = parser.parse_args()
    errors = []
    records = json.loads((ROOT / 'metadata/source_files.json').read_text())['files']
    for rel, record in records.items():
        path = ROOT / rel
        if not path.is_file() or path.is_symlink():
            errors.append('Missing source: ' + rel)
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['backup_sha256']:
            errors.append('Changed backup source: ' + rel)
        if args.compare_live:
            original = Path(record['source'])
            if not original.is_file() or hashlib.sha256(original.read_bytes()).hexdigest() != record['source_sha256']:
                errors.append('Changed/missing live source: ' + rel)
    count = 0
    total = 0
    for path in ROOT.rglob('*'):
        rel = path.relative_to(ROOT)
        if '.git' in rel.parts or any(part.startswith('.snapshot-') for part in rel.parts):
            continue
        if path.is_symlink():
            if path.is_absolute() and not path.resolve().is_relative_to(ROOT):
                errors.append('External symlink: ' + str(rel))
            continue
        if not path.is_file():
            continue
        count += 1
        raw = path.read_bytes()
        total += len(raw)
        if rel.parts[0] in {'data', 'cache', 'logs', 'outputs', 'weights', 'downloads'}:
            errors.append('Runtime artifact: ' + str(rel))
        if (path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name.startswith('.env')
                or path.name in {'hosts.yml', 'config.yaml.local'}):
            errors.append('Forbidden artifact: ' + str(rel))
        if len(raw) > 2 * 1024 * 1024 or b'\0' in raw:
            errors.append('Large/binary file: ' + str(rel))
        for pattern in SECRETS:
            if pattern.search(raw):
                # Never print matching credential content.
                errors.append('Possible credential: ' + str(rel))
                break
    for group in ['scripts', 'tools', 'data_tools', 'tests']:
        for path in (ROOT / group).glob('*.py'):
            try:
                ast.parse(path.read_text(), filename=str(path))
            except SyntaxError:
                errors.append('Invalid Python syntax: ' + str(path.relative_to(ROOT)))
    lock = json.loads((ROOT / 'metadata/sources.lock.json').read_text())
    for repo in lock['repositories']:
        directory = ROOT / repo['path']
        for rel in repo.get('license_files', []):
            if not (ROOT / rel).is_file():
                errors.append('Missing recorded upstream license: ' + rel)
        for alias in repo.get('symlinks', []):
            path = directory / alias['path']
            if not path.is_symlink() or str(path.readlink()) != alias['target']:
                errors.append('Changed/missing compatibility symlink: ' + repo['name'])
        if repo.get('patch'):
            patch = ROOT / repo['patch']
            if not patch.is_file() or hashlib.sha256(patch.read_bytes()).hexdigest() != repo['patch_sha256']:
                errors.append('Patch hash mismatch: ' + repo['name'])
    # Include staged files in checks; an ignored runtime file must not be force-added.
    staged = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-z'])
    tracked = {os.fsdecode(value) for value in staged.split(b'\0') if value}
    if args.staged:
        for rel in sorted(set(records) - tracked):
            errors.append('Source excluded from Git index: ' + rel)
    for value in staged.split(b'\0'):
        if value and not (ROOT / os.fsdecode(value)).exists():
            errors.append('Tracked path missing: ' + os.fsdecode(value))
    print(json.dumps({'files': count, 'bytes': total, 'source_hashes': len(records),
                      'compared_live': args.compare_live, 'errors': errors}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
