"""Copy the audited fragmented-case renders into an independent subset."""
import csv
import hashlib
import json
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


STATS = Path('/root/datasets/MUVA_case_statistics')
SOURCE = Path('/root/datasets/MUVA_case_renders')
DESTINATION = Path('/root/datasets/MUVA_fragmented_dataset')
MANIFEST = STATS / 'multifragment_pairs.csv'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def copy_case(row):
    case_id = row['case_id']
    folder = DESTINATION / case_id
    folder.mkdir(exist_ok=True)
    records = []
    for name in ('reference.jpg', 'target.jpg'):
        source = SOURCE / case_id / name
        target = folder / name
        source_hash = sha256(source)
        if target.exists():
            assert not target.is_symlink() and not target.samefile(source), str(target)
            assert sha256(target) == source_hash, str(target)
        else:
            temporary = folder / (name + '.part')
            shutil.copy2(source, temporary)
            assert sha256(temporary) == source_hash, str(temporary)
            temporary.replace(target)
        assert target.stat().st_size == source.stat().st_size, str(target)
        assert not target.samefile(source), str(target)
        records.append((f'{case_id}/{name}', source_hash, target.stat().st_size))
    assert {p.name for p in folder.iterdir()} == {'reference.jpg', 'target.jpg'}, str(folder)
    return records


def main():
    started = time.monotonic()
    with MANIFEST.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    names = {r['case_id'] for r in rows}
    assert len(rows) == len(names) == 10447
    assert all(int(r['target_components_main']) >= 2 for r in rows)
    assert all(Path(name).name == name for name in names)
    DESTINATION.mkdir(exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(copy_case, row) for row in rows]
        for count, future in enumerate(as_completed(futures), 1):
            records.extend(future.result())
            if count % 1000 == 0 or count == len(rows):
                progress = {'completed_cases': count, 'total_cases': len(rows),
                            'elapsed_seconds': round(time.monotonic() - started)}
                (STATS / 'fragment_subset_progress.json').write_text(json.dumps(progress) + '\n')
                print('COPIED_AND_HASH_VERIFIED', json.dumps(progress), flush=True)

    actual = {p.name for p in DESTINATION.iterdir() if p.is_dir()}
    assert actual == names, (len(actual), len(names))
    assert len(records) == 20894
    shutil.copy2(MANIFEST, DESTINATION / 'pairs.csv')
    assert sha256(MANIFEST) == sha256(DESTINATION / 'pairs.csv')
    with (DESTINATION / 'SHA256SUMS').open('w') as f:
        for relative_path, digest, _ in sorted(records):
            f.write(f'{digest}  {relative_path}\n')

    summary = {
        'dataset_type': 'paired visible-GT overlay images with fragmented targets',
        'source_render_root': str(SOURCE), 'source_pair_manifest': str(MANIFEST),
        'original_muva_root': '/root/datasets/MUVA',
        'case_count': len(rows), 'image_count': len(records),
        'cases_by_split': dict(sorted(Counter(r['split'] for r in rows).items())),
        'selection': 'target_components_main >= 2', 'connectivity': 8,
        'minimum_component_area': 'max(10, ceil(0.01 * target_visible_area))',
        'case_files': ['reference.jpg', 'target.jpg'],
        'total_image_bytes': sum(r[2] for r in records),
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    (DESTINATION / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (DESTINATION / '说明.txt').write_text(
        'MUVA 可见分块案例子集\n\n'
        '本目录是已渲染案例的独立文件副本。共10447组，20894张图片。\n'
        'train：6224组；val：2215组；test：2008组，保留原始划分。\n'
        '每个case_id直接对应一个子文件夹，包含reference.jpg与target.jpg。\n'
        '两张图片分别显示参考视角和目标视角，并叠加同一物体的真实可见掩码。\n'
        '筛选条件：目标存在至少两个显著可见连通分块。采用8邻域，每块面积至少为'
        'max(10, ceil(目标可见面积×1%))像素。该条件描述可见掩码分块。\n'
        '每个目标视角选择同一场景、同一物体的其他可见视角中可见率最高者作为参考。\n\n'
        'pairs.csv：全部案例及原始图片、标注ID、可见率、分块数等对应信息。\n'
        'summary.json：统计口径与数量。\n'
        'SHA256SUMS：每张图片的SHA-256校验值。\n'
        'verification.json：复制后完整性检查结果。\n\n'
        '原案例目录：/root/datasets/MUVA_case_renders\n'
        '原始MUVA图片与标注：/root/datasets/MUVA\n'
    )
    verification = {
        'passed': True, 'case_folders': len(actual), 'images': len(records),
        'every_image_sha256_matches_source': True,
        'independent_files_not_hardlinks_or_symlinks': True,
        'every_case_has_exactly_reference_and_target': True,
        'csv_sha256_matches_source_manifest': True,
        'missing_or_extra_case_folders': 0,
        'elapsed_seconds': round(time.monotonic() - started),
    }
    (DESTINATION / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print('COPY_COMPLETE', json.dumps({'destination': str(DESTINATION), **summary, **verification}), flush=True)


if __name__ == '__main__':
    main()
