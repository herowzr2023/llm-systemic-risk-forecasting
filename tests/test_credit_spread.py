"""The credit spread is the corporate yield premium over government bonds."""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from forecast_data import append_macro_features


class CreditSpreadTest(unittest.TestCase):
    def test_corporate_minus_government_with_missing_release(self):
        dates = pd.date_range('2020-01-01', periods=3)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame({
                '中债国债到期收益率:10年': [3.0, 4.0],
                '中债企业债到期收益率(AAA):10年': [4.5, 3.5],
                '中债国债到期收益率:3月': [1.0, 1.0],
                '中债国债到期收益率:6月': [2.0, 2.0],
            }, index=dates[[0, 2]]).to_csv(root/'macro.csv')
            pd.DataFrame({'银行间同业拆借加权利率:3个月': [2.0]*3}, index=dates).to_csv(root/'credit.csv')
            pd.DataFrame({'Volatility': [0.2]*3, 'Log_Return': [0.01]*3}, index=dates).to_csv(root/'vol.csv')
            out = append_macro_features(pd.DataFrame(index=dates), root/'macro.csv', root/'credit.csv', root/'vol.csv')
            self.assertEqual(out['CS'].tolist(), [1.5, 1.5, -0.5])
            self.assertEqual(out['LS'].tolist(), [1.0]*3)


if __name__ == '__main__':
    unittest.main()
