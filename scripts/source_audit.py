"""Verify the pinned author source before loading a model."""
import subprocess


def verify_source(config):
    repo = config['repo']
    revision = subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip()
    if revision != config['repo_commit']:
        raise RuntimeError(f'Author source revision mismatch: {repo}')
    if subprocess.check_output(['git', '-C', repo, 'diff', 'HEAD', '--'], text=True):
        raise RuntimeError(f'Tracked author source changed: {repo}')
    return revision
