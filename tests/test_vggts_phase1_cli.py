"""CLI contract tests for the VGGT-S phase 1 evaluator entry (no model construction)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_vggts_official import build_parser, validate_run_arguments

CHECKPOINT = str(ROOT / 'outputs/vggts_official/train/muva_seed42_run01/best.pth')
SELECTOR = str(ROOT / 'configs/point_selectors/block_first_v1.json')


def parse(argv):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_run_arguments(parser, args)
    return args


class Phase1CliTests(unittest.TestCase):
    def test_smoke_evaluate_requires_checkpoint(self):
        with self.assertRaises(SystemExit):
            parse(['--mode', 'smoke-evaluate', '--split', 'val'])

    def test_smoke_evaluate_rejects_test_split(self):
        with self.assertRaises(SystemExit):
            parse(['--mode', 'smoke-evaluate', '--split', 'test',
                   '--checkpoint', CHECKPOINT])

    def test_smoke_evaluate_rejects_non_positive_limit(self):
        with self.assertRaises(SystemExit):
            parse(['--mode', 'smoke-evaluate', '--split', 'val',
                   '--checkpoint', CHECKPOINT, '--limit', '0'])

    def test_evaluate_rejects_limit(self):
        with self.assertRaises(SystemExit):
            parse(['--mode', 'evaluate', '--split', 'val', '--checkpoint', CHECKPOINT,
                   '--limit', '8'])

    def test_train_rejects_selector_config(self):
        with self.assertRaises(SystemExit):
            parse(['--mode', 'train', '--selector-config', SELECTOR])

    def test_original_smoke_rejects_checkpoint(self):
        with self.assertRaises(SystemExit):
            parse(['--mode', 'smoke', '--checkpoint', CHECKPOINT])

    def test_resume_is_train_only(self):
        with self.assertRaises(SystemExit):
            parse(['--mode', 'evaluate', '--split', 'val', '--checkpoint', CHECKPOINT,
                   '--resume', CHECKPOINT])

    def test_prepare_points_limit_must_be_positive(self):
        with self.assertRaises(SystemExit):
            parse(['--mode', 'prepare-points', '--limit', '0'])
        args = parse(['--mode', 'prepare-points', '--limit', '4'])
        self.assertEqual(args.limit, 4)

    def test_valid_formal_evaluate(self):
        args = parse(['--mode', 'evaluate', '--split', 'test',
                      '--checkpoint', CHECKPOINT, '--selector-config', SELECTOR])
        self.assertEqual(args.mode, 'evaluate')
        self.assertEqual(args.split, 'test')

    def test_valid_smoke_evaluate_defaults(self):
        args = parse(['--mode', 'smoke-evaluate', '--split', 'val',
                      '--checkpoint', CHECKPOINT])
        self.assertIsNone(args.limit)
        self.assertIsNone(args.selector_config)

    def test_legacy_commands_still_parse(self):
        for argv in (['--mode', 'smoke'],
                     ['--mode', 'check-data'],
                     ['--mode', 'smoke-train'],
                     ['--mode', 'train', '--resume', CHECKPOINT]):
            self.assertEqual(parse(argv).mode, argv[1])


if __name__ == '__main__':
    unittest.main()
