"""Restore locked upstream Git repositories to a NEW vendor directory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run(*args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True, type=Path,
                        help='New/empty directory; never point this at a running experiment')
    args = parser.parse_args()
    out = args.destination.resolve()
    if out == ROOT or ROOT in out.parents or out in ROOT.parents:
        raise SystemExit('Restore outside the backup checkout, into a separate working directory')
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Destination is not empty; refusing to replace existing sources')
    lock = json.loads((ROOT / 'metadata/sources.lock.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    for repo in lock['repositories']:
        revision = repo.get('git_revision')
        if not revision:
            continue  # external/ snapshots are copied separately, without invented history.
        dest = out / repo['name']
        run('git', 'init', str(dest))
        run('git', 'remote', 'add', 'origin', repo['url'], cwd=dest)
        run('git', 'fetch', '--depth', '1', 'origin', revision, cwd=dest)
        run('git', 'checkout', '--detach', 'FETCH_HEAD', cwd=dest)
        if repo.get('patch'):
            patch = ROOT / repo['patch']
            if hashlib.sha256(patch.read_bytes()).hexdigest() != repo['patch_sha256']:
                raise RuntimeError('Patch checksum mismatch')
            run('git', 'apply', '--check', str(patch), cwd=dest)
            run('git', 'apply', str(patch), cwd=dest)
        for alias in repo.get('symlinks', []):
            path = dest / alias['path']
            if path.exists() or path.is_symlink():
                if not path.is_symlink() or str(path.readlink()) != alias['target']:
                    raise RuntimeError('Unexpected compatibility alias: ' + str(path))
            else:
                path.symlink_to(alias['target'])
    print('Restored locked official sources:', out)
    print('Next: restore external snapshots, Conda environments, data and weights; review absolute paths.')


if __name__ == '__main__':
    main()
