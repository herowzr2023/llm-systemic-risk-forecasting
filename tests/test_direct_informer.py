"""Synthetic contract checks only: no training, checkpoint loading, or inference.

Run with the project's scientific Python environment:
    python -m unittest discover -s tests -p test_direct_informer.py -v
"""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INFORMER = ROOT / 'libs' / 'informer'


def cli_contract(arguments):
    """Execute actual CLI parser, direct validation, and setting AST, without Exp.

    The excluded entry-point statements create/train/evaluate the model. Parser
    defaults and setting expressions are read from production, never duplicated.
    """
    tree = ast.parse((INFORMER / 'main_informer.py').read_text(encoding='utf-8'))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(name in {'parser', 'args', 'setting'} for name in names):
                nodes.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if func.value.id == 'parser' and func.attr == 'add_argument':
                    nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == 'infer_custom_dimensions':
            nodes.append(node)
        elif isinstance(node, ast.If) and ast.unparse(node.test) == "args.data == 'direct'":
            nodes.append(node)
    namespace = {'argparse': argparse, 'os': os, 'pd': pd}
    with patch.object(sys, 'argv', ['main_informer.py', *arguments]):
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(INFORMER / 'main_informer.py'), 'exec'), namespace)
    return namespace


class ConfigContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('direct_batch_contract', ROOT / 'code' / '06_batch_train_informer.py')
        cls.batch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.batch)

    def test_all_active_jobs_scalar_direct_and_cli_setting_match(self):
        raw = json.loads((ROOT / 'configs' / '06_informer_batch.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'synthetic.csv'
            pd.DataFrame({'date': ['2020-01-01'], 'feature': [3.], 'target': [4.]}).to_csv(source, index=False)
            for job in raw['jobs']:
                for variant in self.batch.variants_for_job(job):
                    args = self.batch.effective_args(raw, job, variant, 'synthetic', 2)
                    with self.subTest(job=job['name'], variant=variant['name']):
                        self.assertEqual(args['data'], 'direct')
                        self.assertEqual(args['pred_len'], 1)
                        self.assertEqual(args['features'], 'MS')
                        self.assertEqual(args['lead'], 5 if '5d' in job['name'] else 1)
                        for key in ['results_dir', 'checkpoints']:
                            self.assertIn('direct_h1_h5', args[key])
                        modified = dict(raw, common_args=dict(raw['common_args'], root_path=directory))
                        test_variant = dict(variant, data_path='synthetic.csv', predict_col='target')
                        command = self.batch.command_for_job(modified, job, test_variant, 'synthetic', 2)
                        namespace = cli_contract(command[2:])
                        self.assertEqual(namespace['args'].c_out, 1)
                        self.assertEqual(namespace['args'].enc_in, 2)
                        self.assertEqual(namespace['setting'], self.batch.setting_for_job(modified, job, test_variant, 'synthetic', 2))

    def test_single_config_direct(self):
        args = json.loads((ROOT / 'configs' / '06_informer_single.json').read_text(encoding='utf-8'))['args']
        self.assertEqual((args['data'], args['pred_len'], args['lead'], args['features']), ('direct', 1, 1, 'MS'))
        self.assertTrue(args['calendar_path'])
        self.assertIn('direct_h1_h5', args['results_dir'])
        self.assertIn('direct_h1_h5', args['checkpoints'])

    def test_direct_cli_column_subset_dimensions_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            pd.DataFrame({'date': ['2020-01-01'], 'feature': [3.], 'unused': [7.], 'target': [4.]}).to_csv(
                Path(directory) / 'source.csv', index=False)
            base = ['--model', 'informer', '--data', 'direct', '--features', 'MS',
                    '--calendar_path', 'unused.csv', '--root_path', directory,
                    '--path', 'source.csv', '--predict_col', 'target']
            namespace = cli_contract(base + ['--cols', 'feature', 'target'])
            self.assertEqual((namespace['args'].enc_in, namespace['args'].dec_in, namespace['args'].c_out), (2, 2, 1))
            for cols in [['unknown', 'target'], ['feature', 'feature', 'target'], ['feature'], ['date', 'target']]:
                with self.subTest(cols=cols), self.assertRaises(SystemExit) as error:
                    cli_contract(base + ['--cols', *cols])
                self.assertEqual(error.exception.code, 2)

    def test_cli_rejects_unseen_predict_and_joint_output(self):
        base = ['--model', 'informer', '--data', 'direct', '--features', 'MS', '--calendar_path', 'unused.csv']
        for extra in [['--do_predict'], ['--pred_len', '5'], ['--features', 'M'], ['--is_class']]:
            with self.subTest(extra=extra), self.assertRaises(SystemExit) as error:
                cli_contract(base + extra)
            self.assertEqual(error.exception.code, 2)


class DirectDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(INFORMER))
        from data.data_loader import Dataset_Direct
        cls.dataset_class = Dataset_Direct

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.dates = pd.bdate_range('2020-01-01', periods=100)
        self.raw = pd.DataFrame({'date': self.dates, 'feature': np.arange(100) * 2., 'target': np.arange(100) + 100.})
        self.calendar = self.directory / 'calendar.csv'
        pd.DataFrame({'date': self.dates}).to_csv(self.calendar, index=False)

    def dataset(self, raw=None, flag='train', **overrides):
        (self.raw if raw is None else raw).to_csv(self.directory / 'source.csv', index=False)
        args = dict(root_path=str(self.directory), data_path='source.csv', calendar_path=str(self.calendar),
                    flag=flag, size=[4, 2, 1], features='MS', target='target', lead=5,
                    inverse=True, scaler_type='minmax', freq='d', timeenc=0)
        args.update(overrides)
        return self.dataset_class(**args)

    def test_exact_t_plus_five_scalar_label_and_observed_prefix(self):
        for flag in ['train', 'val', 'test']:
            dataset = self.dataset(flag=flag)
            self.assertGreater(len(dataset), 0)
            for i, (origin, target) in enumerate(dataset.windows):
                x, y, x_mark, y_mark, dates = dataset[i]
                self.assertEqual(target, origin + 5)
                self.assertEqual(y.shape, (3, 2))
                self.assertEqual(y[-1, -1:].shape, (1,))
                self.assertEqual(y[-1, -1], self.raw.loc[target, 'target'])
                np.testing.assert_array_equal(y[:2], dataset.data_x[origin - 1:origin + 1])
                self.assertEqual(y[-1, 0], 0)
                self.assertEqual(pd.Timestamp(dates[-2]), self.dates[origin])
                self.assertEqual(pd.Timestamp(dates[-1]), self.dates[target])
                self.assertEqual((x.shape, len(x_mark), len(y_mark)), ((4, 2), 4, 3))
                expected_split = 0 if target < 70 else 1 if target < 80 else 2
                self.assertEqual(dataset.set_type, expected_split)
                if flag == 'val':
                    self.assertGreaterEqual(origin, 69)
                elif flag == 'test':
                    self.assertGreaterEqual(origin, 79)

    def test_missing_rows_use_calendar_lead_and_skip_gapped_history(self):
        raw = self.raw.drop(index=[20, 23, 31]).reset_index(drop=True)
        dataset = self.dataset(raw)
        positions = {date: i for i, date in enumerate(self.dates)}
        for origin, target in dataset.windows:
            history = [positions[d] for d in raw.loc[origin - 3:origin, 'date']]
            self.assertEqual(history, list(range(history[-1] - 3, history[-1] + 1)))
            self.assertEqual(positions[raw.loc[target, 'date']] - positions[raw.loc[origin, 'date']], 5)
        # Origin 19 has calendar target 24, even though only three retained rows separate them.
        origin = raw.index[raw['date'] == self.dates[19]][0]
        target = raw.index[raw['date'] == self.dates[24]][0]
        self.assertIn((origin, target), dataset.windows)
        self.assertNotEqual(target - origin, 5)
        # Calendar target 23 is absent, so origin 18 must not receive a substitute label.
        self.assertNotIn(raw.index[raw['date'] == self.dates[18]][0], [o for o, _ in dataset.windows])

    def test_scaling_fit_uses_train_rows_only(self):
        baseline = self.dataset()
        changed = self.raw.copy()
        changed.loc[70:, ['feature', 'target']] += 100000.
        for flag in ['train', 'val', 'test']:
            dataset = self.dataset(changed, flag=flag)
            np.testing.assert_array_equal(dataset.data_x[:70], baseline.data_x[:70])
            self.assertGreater(dataset.data_x[70, -1], 100.)
        self.assertAlmostEqual(float(baseline.data_x[69, -1]), 1.)
        scaled = self.dataset(inverse=False)
        _, target = scaled.windows[0]
        self.assertAlmostEqual(float(scaled[0][1][-1, -1]), target / 69., places=6)
        unscaled = self.dataset(scale=False)
        np.testing.assert_array_equal(unscaled.data_x, self.raw[['feature', 'target']].to_numpy(dtype=np.float32))
        sample = np.array([1., 2.])
        np.testing.assert_array_equal(unscaled.inverse_transform_y(sample), sample)

    def test_rejects_joint_horizon_wrong_order_and_nonfinite_input(self):
        for kwargs in [dict(size=[4, 2, 5]), dict(features='M'), dict(lead=2), dict(is_class=True), dict(calendar_path=None)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.dataset(**kwargs)
        for raw in [self.raw.iloc[::-1], pd.concat([self.raw, self.raw.iloc[-1:]]), self.raw.assign(target=np.nan)]:
            with self.assertRaises(ValueError):
                self.dataset(raw)


class DirectExportTests(unittest.TestCase):
    def test_production_export_scalar_provenance_and_count_checks(self):
        # Load the exact standalone production helper without importing plotting/model dependencies.
        tree = ast.parse((INFORMER / 'exp' / 'exp_informer.py').read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'direct_result_frame')
        namespace = {'np': np, 'pd': pd}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'exp_informer.py', 'exec'), namespace)
        export = namespace['direct_result_frame']
        pred = np.array([[[2.]], [[1.]]])
        true = np.array([[[4.]], [[3.]]])
        origins, targets = ['2020-01-02', '2020-01-01'], ['2020-01-09', '2020-01-08']
        frame = export(pred, true, origins, targets, 5)
        self.assertEqual(frame['pred'].tolist(), [1., 2.])
        self.assertEqual(frame['actual'].tolist(), [3., 4.])
        self.assertTrue((frame['origin_date'] == frame['origin']).all())
        self.assertTrue((frame['date'] == frame['target_date']).all())
        self.assertEqual(frame['forecast_mode'].tolist(), ['direct', 'direct'])
        self.assertEqual(frame['horizon'].tolist(), [5, 5])
        self.assertEqual(frame['lead'].tolist(), [5, 5])
        for values in [(np.zeros((2, 5, 1)), true, origins, targets, 5),
                       (pred, true, origins[:1], targets, 5),
                       (pred, true, origins[:1] * 2, targets, 5)]:
            with self.assertRaises(ValueError):
                export(*values)


if __name__ == '__main__':
    unittest.main()
