"""Verify every paired-render folder without running any learned model."""
import json
import multiprocessing as mp
from pathlib import Path
from PIL import Image

ROOT = Path('/root/datasets/MUVA_case_renders')
STATS = Path('/root/datasets/MUVA_case_statistics')


def check_case(job):
    index, case_id = job
    folder = ROOT / case_id
    assert {p.name for p in folder.iterdir()} == {'reference.jpg', 'target.jpg'}, str(folder)
    total = 0
    for name in ['reference.jpg', 'target.jpg']:
        p = folder / name
        assert p.is_file() and not p.is_symlink(), str(p)
        size = p.stat().st_size
        assert size > 1000, str(p)
        with p.open('rb') as f:
            assert f.read(2) == b'\xff\xd8', str(p)
            f.seek(-2, 2)
            assert f.read(2) == b'\xff\xd9', str(p)
        with Image.open(p) as image:
            assert image.format == 'JPEG' and image.size == (1920, 1170), str(p)
            image.verify()
        # A reproducible spread through all splits receives full pixel decoding.
        if index % 1000 == 0:
            with Image.open(p) as image:
                image.load()
        total += size
    return total, int(index % 1000 == 0) * 2


def main():
    with (STATS / 'pairs.jsonl').open() as f:
        names = [json.loads(line)['case_id'] for line in f]
    actual = {p.name for p in ROOT.iterdir()}
    assert actual == set(names), (len(actual), len(names), list(actual-set(names))[:5])
    total = decoded = count = 0
    with mp.Pool(12) as pool:
        for size, ndecoded in pool.imap_unordered(check_case, enumerate(names), chunksize=100):
            total += size; decoded += ndecoded; count += 1
            if count % 20000 == 0:
                print('VERIFIED_CASES', count, '/', len(names), flush=True)
    report = {'passed': True, 'verified_case_folders': count, 'verified_images': count * 2,
              'each_folder_contains_only_reference_and_target_jpg': True,
              'all_jpeg_start_and_end_markers_valid': True,
              'all_jpeg_headers_and_dimensions_valid': True,
              'full_pixel_decoding_sample_images': decoded,
              'total_rendered_bytes': total, 'unexpected_or_missing_folders': 0}
    (STATS / 'render_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
