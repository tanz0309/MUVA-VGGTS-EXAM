"""Copy an allowlisted source snapshot into this backup, never into live training."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {
    '.py', '.pyi', '.sh', '.bash', '.zsh', '.json', '.yaml', '.yml', '.toml',
    '.ini', '.cfg', '.conf', '.txt', '.md', '.rst', '.c', '.cc', '.cpp', '.cu',
    '.cuh', '.h', '.hpp', '.cmake', '.in', '.html', '.css', '.js', '.ts',
    '.bat', '.ps1', '.cff', '.bib',
}
SPECIAL_NAMES = {'Dockerfile', 'Makefile', '.gitignore', '.gitattributes', '.gitmodules'}
SKIP_PARTS = {'.git', '__pycache__', 'build', 'dist', 'node_modules', '.venv', '.idea', '.vscode'}
UPSTREAMS = {
    'INSID3-official': 'https://github.com/visinf/INSID3.git',
    'O-MaMa-official': 'https://github.com/Maria-SanVil/O-MaMa.git',
    'FastSAM-official': 'https://github.com/CASIA-IVA-Lab/FastSAM.git',
    'VGGT-S-official': 'https://github.com/buaa-colalab/VGGT-S.git',
    'vggt-official': 'https://github.com/facebookresearch/vggt.git',
    'V2-SAM-official': 'https://github.com/jaychempan/V2-SAM.git',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(repo, *args, binary=False):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=not binary)


def allowed(path):
    return (not any(p in SKIP_PARTS or p.endswith('.egg-info') for p in path.parts)
            and not path.name.startswith('.env')
            and (path.suffix.lower() in SOURCE_SUFFIXES or path.name in SPECIAL_NAMES
                 or path.name.upper().startswith(('LICENSE', 'NOTICE', 'COPYRIGHT'))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path('/root/datasets/MUVA_experiments'))
    parser.add_argument('--migration-root', type=Path, default=Path('/root/to_migrate/_migration'))
    parser.add_argument('--dataset-root', type=Path, default=Path('/root/datasets'))
    parser.add_argument('--apply', action='store_true', help='Write the reviewed snapshot into this backup')
    args = parser.parse_args()
    project = args.project_root.resolve()
    if project == ROOT or project in ROOT.parents or ROOT in project.parents:
        raise SystemExit('Backup and live project must be independent directories')
    if not (project / 'scripts/run_experiment.py').is_file():
        raise SystemExit('Live project entry is missing')
    if not args.apply:
        print(json.dumps({'source': str(project), 'backup': str(ROOT),
                          'action': 'Read-only plan. Use --apply to synchronize the source allowlist.'}))
        return

    records = {}
    repos = []
    source_checks = {}
    managed = ('scripts', 'configs', 'tests', 'vendor', 'external', 'environments',
               'data_tools', 'patches', 'metadata')
    with tempfile.TemporaryDirectory(prefix='.snapshot-', dir=ROOT) as tmp:
        stage = Path(tmp)

        def put(source, target, transform=False):
            source = Path(source)
            if not source.is_file() or source.is_symlink():
                raise RuntimeError('Expected regular source file: ' + str(source))
            original = source.read_bytes()
            if len(original) > 2 * 1024 * 1024 or b'\0' in original:
                raise RuntimeError('Unexpected large/binary source: ' + str(source))
            data = original
            if transform:
                text = data.decode('utf-8')
                prefix = str(project) + '/'
                def link(match):
                    relative = os.path.relpath(match.group(1)[len(prefix):], Path(target).parent)
                    return '](' + relative + ')'
                text = re.sub(r'\]\((' + re.escape(prefix) + r'[^)]+)\)', link, text)
                data = text.encode('utf-8')
            dst = stage / target
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            dst.chmod(0o755 if os.access(source, os.X_OK) else 0o644)
            records[str(target)] = {'source': str(source), 'source_sha256': sha(original),
                                    'backup_sha256': sha(data)}
            source_checks[source] = sha(original)

        for name, suffix in [('scripts', '*.py'), ('configs', '*.json'), ('tests', '*.py'), ('docs', '*.md')]:
            for source in sorted((project / name).glob(suffix)):
                put(source, str(source.relative_to(project)), transform=(name == 'docs'))
        put(project / 'run.sh', 'run.sh')
        put(project / 'README.md', 'docs/SERVER_README.md', transform=True)

        for name, url in UPSTREAMS.items():
            repo = project / 'vendor' / name
            revision = git(repo, 'rev-parse', 'HEAD').strip()
            patch = git(repo, 'diff', 'HEAD', '--', binary=True)
            names = git(repo, 'ls-files', '-z', binary=True).split(b'\0')
            copied = []
            omitted = []
            for value in names:
                if not value:
                    continue
                rel = Path(os.fsdecode(value))
                source = repo / rel
                if source.is_file() and not source.is_symlink() and allowed(rel):
                    target = f'vendor/{name}/{rel.as_posix()}'
                    put(source, target)
                    copied.append(target)
                else:
                    omitted.append(rel.as_posix())
            info = {'name': name, 'path': f'vendor/{name}', 'source': str(repo),
                    'url': url, 'git_revision': revision, 'copied_files': len(copied),
                    'omitted_non_source_files': omitted, 'patch': None, 'symlinks': [],
                    'license_files': [p for p in copied if Path(p).name.upper().startswith('LICENSE')],
                    'root_license_files': [p for p in copied if Path(p).parent == Path(f'vendor/{name}')
                                           and Path(p).name.upper().startswith('LICENSE')]}
            if patch:
                patch_path = f'patches/{name}.patch'
                (stage / 'patches').mkdir(exist_ok=True)
                (stage / patch_path).write_bytes(patch)
                info['patch'] = patch_path
                info['patch_sha256'] = sha(patch)
            if name == 'V2-SAM-official':
                alias = repo / 'projects/v2sam'
                if not alias.is_symlink() or str(alias.readlink()) != 'v2sam_fusion':
                    raise RuntimeError('Unexpected V2-SAM compatibility alias')
                target = stage / f'vendor/{name}/projects/v2sam'
                target.symlink_to('v2sam_fusion')
                info['symlinks'] = [{'path': 'projects/v2sam', 'target': 'v2sam_fusion'}]
            # Recheck after copying; a concurrent change must not produce mixed source state.
            if git(repo, 'rev-parse', 'HEAD').strip() != revision or git(repo, 'diff', 'HEAD', '--', binary=True) != patch:
                raise RuntimeError('Official source changed while copying: ' + name)
            repos.append(info)

        external = [
            ('dinov2-runtime', Path('/root/.cache/torch/hub/facebookresearch_dinov2_main'), 'https://github.com/facebookresearch/dinov2'),
            ('dinov3-runtime', Path('/root/.cache/torch/hub/facebookresearch_dinov3_main'), 'https://github.com/facebookresearch/dinov3'),
            ('mmengine-runtime', Path('/root/V2-SAM/mmengine'), 'https://github.com/open-mmlab/mmengine'),
        ]
        for name, source_root, url in external:
            count = 0
            for source in sorted(source_root.rglob('*')):
                if source.is_file() and not source.is_symlink() and allowed(source.relative_to(source_root)):
                    put(source, f'external/{name}/{source.relative_to(source_root).as_posix()}')
                    count += 1
            if not count:
                raise RuntimeError('Required dependency source is empty: ' + str(source_root))
            repos.append({'name': name, 'path': f'external/{name}', 'source': str(source_root),
                          'url': url, 'git_revision': None, 'copied_files': count,
                          'license_files': [p for p in records if p.startswith(f'external/{name}/')
                                            and Path(p).name.upper().startswith('LICENSE')],
                          'provenance': 'Existing runtime source snapshot without Git metadata'})

        for source in sorted((args.migration_root / 'environments').rglob('*')):
            if source.is_file() and allowed(source.relative_to(args.migration_root)):
                put(source, 'environments/' + source.relative_to(args.migration_root / 'environments').as_posix())
        for source in sorted((args.dataset_root / 'MUVA_case_statistics').glob('*.py')):
            put(source, 'data_tools/' + source.name)
        put(args.migration_root / 'huggingface/restore.py', 'data_tools/restore_hf_data.py')
        put(args.migration_root / 'huggingface/SHA256SUMS', 'data_tools/HF_SHA256SUMS')

        # Verify every copied source, including live adapters, stayed stable during the snapshot.
        for source, expected in source_checks.items():
            if sha(source.read_bytes()) != expected:
                raise RuntimeError('Source changed while copying: ' + str(source))
        meta = stage / 'metadata'
        meta.mkdir(parents=True, exist_ok=True)
        def write(name, data):
            (meta / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        write('sources.lock.json', {'format_version': 1, 'repositories': repos})
        write('source_files.json', {'format_version': 1, 'files': records})
        write('dataset.json', {
            'repository': 'tzz2002/MUVA', 'repo_type': 'dataset', 'private': True,
            'revision': '5c027b9917fe0373ca80f5b5dea5c317edf34603', 'folder': 'core-20261109',
            'cases': {'train': 6224, 'val': 2215, 'test': 2008},
            'unique_rgb': {'train': 6778, 'val': 2371, 'test': 2283},
            'manifest_sha256': {split: sha((project / f'data/manifests/{split}.jsonl').read_bytes())
                                for split in ('train', 'val', 'test')},
            'data_in_git': False,
        })
        write('backup.json', {
            'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'live_project': str(project), 'backup_worktree': str(ROOT), 'source_file_count': len(records),
            'gpu': 'RTX 4090; driver-reported 49140 MiB', 'seed': 42,
            'note': 'Code-only snapshot; source hashes verified. Not a model training or accuracy result.',
        })
        for name in managed:
            dst = ROOT / name
            if dst.exists():
                if dst.is_symlink():
                    raise RuntimeError('Refusing to replace managed symlink: ' + str(dst))
                shutil.rmtree(dst)
            src = stage / name
            if src.exists():
                shutil.move(str(src), str(dst))
        (stage / 'run.sh').replace(ROOT / 'run.sh')
        docs = ROOT / 'docs'
        docs.mkdir(exist_ok=True)
        # Keep backup-maintained documents alongside the synchronized experiment documents.
        for old in docs.glob('[0-9][0-9]_*.md'):
            old.unlink()
        for source in (stage / 'docs').glob('*.md'):
            source.replace(docs / source.name)
    print(json.dumps({'result': 'synchronized', 'backup': str(ROOT), 'source_files': len(records),
                      'repositories': len(repos)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
