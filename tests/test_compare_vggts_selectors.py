"""Tests for the saved-result A/B comparator (no model, no GPU)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from common import summarize
import compare_vggts_selectors as compare

HASH_A = {'candidate_points_sha256': 'cand', 'target_crop_rgb_sha256': 'tgt',
          'source_crop_rgb_sha256': 'src', 'source_crop_mask_sha256': 'mask',
          'meta_sha256': 'meta'}


def row(case_id, **values):
    base = {'case_id': case_id, 'split': 'val', 'components': 2, 'group': 'upright',
            'target_visibility': 0.5, 'inference_seconds': 0.1,
            'peak_allocated_gpu_mib': 1.0, 'failed': False, 'error': None,
            'visible_iou': .5, 'dice': .5, 'boundary_f': .5, 'cr50': .5,
            'scr50': .5, 'all_components_covered50': False,
            'cr70': .4, 'scr70': .4, 'all_components_covered70': False,
            'precision': .5, 'recall': .5, 'empty_prediction': False}
    base.update(values)
    return base


def make_run(root, name, method, rows, *, scope='full_split', status='complete',
             checkpoint='ckpt-a', manifests=None, config=None, metric='metric-a',
             audits=None):
    directory = Path(root) / name
    directory.mkdir()
    run_config = {'point_selector': {'effective_config': {'method': method, 'budget': 5}},
                  'evaluation_checkpoint': {'sha256': checkpoint},
                  'manifests': manifests or {'val': 'manifest-a'},
                  'config': config or {'protocol': 'v1'},
                  'metric_implementation_sha256': metric}
    (directory / 'run_config.json').write_text(json.dumps(run_config))
    (directory / 'status.json').write_text(json.dumps({'status': status}))
    summary = summarize(rows)
    summary['scope'] = scope
    (directory / 'summary.json').write_text(json.dumps(summary))
    with (directory / 'per_case.jsonl').open('w') as handle:
        for item in rows:
            handle.write(json.dumps(item) + '\n')
    prompts = directory / 'prompts'
    prompts.mkdir()
    for case_id, audit in (audits or {}).items():
        (prompts / f'{case_id}.json').write_text(json.dumps(audit))
    return directory


def audit(inputs=None, fallback=None):
    selector = {'method_requested': 'block_first', 'method_effective':
                'kmeans' if fallback else 'block_first',
                'fallback_reason': fallback, 'clusters': [], 'selector_seconds': 0.01,
                'unique_candidate_count': 40, 'noise_count_unique': 0}
    return {'selection_inputs': dict(inputs or HASH_A), 'point_selector': selector}


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def pair(self, **overrides):
        base_rows = [row('a', visible_iou=.5, dice=.4), row('b', visible_iou=.7)]
        cand_rows = [row('a', visible_iou=.6, dice=.3), row('b', visible_iou=.7)]
        base = make_run(self.root, 'base', 'kmeans', base_rows,
                        audits={'a': audit(), 'b': audit()}, **overrides.get('base', {}))
        cand = make_run(self.root, 'cand', 'block_first', cand_rows,
                        audits={'a': audit(), 'b': audit()}, **overrides.get('cand', {}))
        return compare.load_run(base), compare.load_run(cand)

    def test_metric_delta_and_percentage_points(self):
        metrics, _b, _c = compare.compare_metrics(
            [row('a', visible_iou=.5, dice=.4)], [row('a', visible_iou=.6, dice=.3)])
        self.assertAlmostEqual(metrics['visible_iou']['delta'], .1)
        self.assertAlmostEqual(metrics['visible_iou']['delta_pp'], 10.)
        self.assertAlmostEqual(metrics['dice']['delta'], -.1)
        self.assertAlmostEqual(metrics['dice']['delta_pp'], -10.)

    def test_failed_rows_remain_in_denominator(self):
        metrics, _b, _c = compare.compare_metrics(
            [row('a', visible_iou=.8), row('b', visible_iou=0., failed=True)],
            [row('a', visible_iou=.8), row('b', visible_iou=0., failed=True)])
        self.assertAlmostEqual(metrics['visible_iou']['baseline'], .4)

    def test_reversed_or_same_method_directories_are_rejected(self):
        base = make_run(self.root, 'base', 'block_first', [row('a')])
        cand = make_run(self.root, 'cand', 'kmeans', [row('a')])
        with self.assertRaises(ValueError):
            compare.validate_pair(compare.load_run(base), compare.load_run(cand))
        same = make_run(self.root, 'same', 'kmeans', [row('a')])
        with self.assertRaises(ValueError):
            compare.validate_pair(compare.load_run(base), compare.load_run(same))

    def test_shuffled_order_is_aligned_and_noted(self):
        base, cand = self.pair()
        # Rewrite candidate rows in reverse order.
        cand_dir = Path(cand['run_dir'])
        rows = [cand['row_by_id']['b'], cand['row_by_id']['a']]
        with (cand_dir / 'per_case.jsonl').open('w') as handle:
            for item in rows:
                handle.write(json.dumps(item) + '\n')
        cand = compare.load_run(cand_dir)
        validation = compare.validate_pair(base, cand)
        self.assertFalse(validation['same_order'])
        self.assertTrue(any('align' in note for note in validation['notes']))

    def test_duplicate_ids_are_rejected(self):
        directory = make_run(self.root, 'dup', 'kmeans', [row('a'), row('a')])
        with self.assertRaises(ValueError):
            compare.load_run(directory)

    def test_missing_sample_is_rejected(self):
        base = make_run(self.root, 'base', 'kmeans', [row('a'), row('b')])
        cand = make_run(self.root, 'cand', 'block_first', [row('a')])
        with self.assertRaises(ValueError):
            compare.validate_pair(compare.load_run(base), compare.load_run(cand))

    def test_checkpoint_manifest_config_and_metric_mismatches_are_rejected(self):
        base = make_run(self.root, 'base', 'kmeans', [row('a')])
        for name, kwargs in [('ckpt', {'checkpoint': 'other'}),
                             ('manifest', {'manifests': {'val': 'other'}}),
                             ('config', {'config': {'protocol': 'v2'}}),
                             ('metric', {'metric': 'metric-b'})]:
            cand = make_run(self.root, 'cand-' + name, 'block_first', [row('a')], **kwargs)
            with self.assertRaises(ValueError, msg=name):
                compare.validate_pair(compare.load_run(base), compare.load_run(cand))

    def test_different_candidate_or_crop_hashes_are_rejected(self):
        base_rows = [row('a')]
        base = make_run(self.root, 'base', 'kmeans', base_rows, audits={'a': audit()})
        bad = dict(HASH_A)
        bad['candidate_points_sha256'] = 'different'
        cand = make_run(self.root, 'cand', 'block_first', base_rows,
                        audits={'a': audit(bad)})
        with self.assertRaises(ValueError):
            compare.validate_pair(compare.load_run(base), compare.load_run(cand))

    def test_smoke_requires_the_flag_and_cannot_mix_with_full(self):
        smoke = make_run(self.root, 'smoke', 'kmeans', [row('a')],
                         scope='smoke_only_not_benchmark')
        with self.assertRaises(ValueError):
            compare.load_run(smoke)
        loaded = compare.load_run(smoke, allow_smoke=True)
        self.assertEqual(loaded['scope'], 'smoke_only_not_benchmark')
        full = make_run(self.root, 'full', 'block_first', [row('a')])
        with self.assertRaises(ValueError):
            compare.validate_pair(loaded, compare.load_run(full), allow_smoke=True)

    def test_interrupted_run_is_rejected(self):
        directory = make_run(self.root, 'broken', 'kmeans', [row('a')],
                             status='incomplete_failed')
        with self.assertRaises(ValueError):
            compare.load_run(directory)

    def test_failures_mark_comparison_unclean(self):
        base = make_run(self.root, 'base', 'kmeans', [row('a')],
                        audits={'a': audit()})
        cand = make_run(self.root, 'cand', 'block_first',
                        [row('a', failed=True, visible_iou=0.)])
        validation = compare.validate_pair(compare.load_run(base), compare.load_run(cand))
        self.assertFalse(validation['valid_clean_comparison'])
        self.assertEqual(validation['failures']['block_first'], 1)
        self.assertGreater(validation['audit_unavailable'], 0)

    def test_fallback_samples_stay_in_main_table(self):
        base = make_run(self.root, 'base', 'kmeans', [row('a'), row('b')],
                        audits={'a': audit(), 'b': audit()})
        cand = make_run(self.root, 'cand', 'block_first', [row('a'), row('b')],
                        audits={'a': audit(fallback='all_noise'), 'b': audit()})
        loaded = compare.load_run(cand)
        summary = compare._selector_summary(loaded)
        self.assertEqual(summary['fallback']['numerator'], 1)
        self.assertEqual(summary['fallback']['denominator'], 2)
        self.assertEqual(summary['total_evaluated_samples'], 2)
        validation = compare.validate_pair(compare.load_run(base), loaded)
        self.assertTrue(validation['valid_clean_comparison'])
        metrics, _b, _c = compare.compare_metrics(
            [loaded['row_by_id']['a'], loaded['row_by_id']['b']],
            [loaded['row_by_id']['a'], loaded['row_by_id']['b']])
        self.assertIn('visible_iou', metrics)

    def test_write_comparison_creates_all_outputs(self):
        base, cand = self.pair()
        validation = compare.validate_pair(base, cand)
        ids = sorted(base['row_by_id'])
        metrics, _b, _c = compare.compare_metrics(
            [base['row_by_id'][case_id] for case_id in ids],
            [cand['row_by_id'][case_id] for case_id in ids])
        output = Path(self.root) / 'comparison'
        compare.write_comparison(output, base, cand, metrics, validation,
                                 [base['row_by_id'][case_id] for case_id in ids],
                                 [cand['row_by_id'][case_id] for case_id in ids])
        for name in ('comparison.csv', 'comparison.json', 'paired_deltas.csv',
                     'comparison.md', 'selector_summary.json'):
            self.assertTrue((output / name).is_file(), name)
        payload = json.loads((output / 'comparison.json').read_text())
        self.assertTrue(payload['valid_clean_comparison'])
        self.assertIn('Block-First', (output / 'comparison.md').read_text())


if __name__ == '__main__':
    unittest.main()
