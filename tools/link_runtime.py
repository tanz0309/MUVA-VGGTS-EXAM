"""Link existing runtime storage into the canonical project; never move live runs."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_NAMES = ('data', 'cache', 'logs', 'outputs', 'weights')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/root/datasets/MUVA_experiments'))
    parser.add_argument('--apply', action='store_true', help='Create missing links after all checks pass')
    args = parser.parse_args()
    source = args.source.resolve()
    if source == ROOT or ROOT in source.parents or source in ROOT.parents:
        raise SystemExit('Runtime storage must be independent of the project')
    plan = []
    for name in RUNTIME_NAMES:
        target, link = source / name, ROOT / name
        if not target.is_dir():
            raise SystemExit('Required runtime directory is missing: ' + str(target))
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != target.resolve():
                raise SystemExit('Refusing to replace existing project path: ' + str(link))
            action = 'already linked'
        else:
            action = 'create link' if args.apply else 'would create link'
        plan.append({'name': name, 'target': str(target), 'action': action})
    if args.apply:
        for item in plan:
            if item['action'] == 'create link':
                (ROOT / item['name']).symlink_to(item['target'], target_is_directory=True)
    print(json.dumps({'project': str(ROOT), 'applied': args.apply, 'links': plan},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
