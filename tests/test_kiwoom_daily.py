import unittest
from unittest.mock import Mock, patch
import pandas as pd
from src.data.kiwoom_daily import fetch_daily, parse_daily, audit, KiwoomAPIError


def bar(dt, price):
    return {"dt": dt, "open_pric": str(price), "high_pric": str(price + 10), "low_pric": str(price - 10), "cur_prc": str(price), "trde_qty": "123", "upd_stkpc_tp": "", "upd_rt": ""}


class TestKiwoomDaily(unittest.TestCase):
    def test_parse_and_sort(self):
        df = parse_daily([bar("20180504", 53000), bar("20180427", 52000)])
        self.assertEqual(df.index.min(), pd.Timestamp("2018-04-27"))
        self.assertEqual(df.iloc[0]["Close"], 52000)

    def test_audit(self):
        df = parse_daily([bar("20180504", 53000)])
        self.assertFalse(audit(df)["InvalidOHLC"].iloc[0])

    @patch("src.data.kiwoom_daily.time.sleep")
    def test_pagination(self, sleep):
        session = Mock()
        first = Mock(status_code=200, headers={"cont-yn": "Y", "next-key": "abc"})
        first.json.return_value = {"return_code": 0, "stk_dt_pole_chart_qry": [bar("20261110", 100)]}
        second = Mock(status_code=200, headers={"cont-yn": "N"})
        second.json.return_value = {"return_code": 0, "stk_dt_pole_chart_qry": [bar("20180101", 80)]}
        session.post.side_effect = [first, second]
        df = fetch_daily(session, "https://example.test", "fake", "005930", "2018-01-01", "2026-11-12", delay=1.1)
        self.assertEqual(len(df), 2)
        self.assertEqual(session.post.call_args_list[1].kwargs["headers"]["next-key"], "abc")
        sleep.assert_called_once()

    def test_duplicate_conflict(self):
        with self.assertRaises(KiwoomAPIError):
            parse_daily([bar("20180504", 53000), bar("20180504", 54000)])


    def test_zero_volume_with_positive_prices(self):
        """OHLC가 양수여도 거래량 0이면 매매 불가능 후보."""
        row = bar("20180430", 53000)
        row["trde_qty"] = "0"

        df = parse_daily([row])
        result = audit(df)

        self.assertTrue(result["NonTradingCandidate"].iloc[0])
        self.assertTrue(result["SuspensionCandidate"].iloc[0])
        self.assertFalse(result["Tradable"].iloc[0])
        self.assertFalse(result["InvalidOHLC"].iloc[0])

    def test_normal_trading_day(self):
        """정상 OHLCV는 매매 가능 후보."""
        df = parse_daily([bar("20180504", 53000)])
        result = audit(df)

        self.assertTrue(result["Tradable"].iloc[0])
        self.assertFalse(result["NonTradingCandidate"].iloc[0])
        self.assertFalse(result["InvalidOHLC"].iloc[0])

    def test_invalid_ohlc(self):
        """종가가 고가보다 높으면 가격 오류."""
        row = bar("20180504", 53000)
        row["high_pric"] = "52900"

        df = parse_daily([row])
        result = audit(df)

        self.assertTrue(result["InvalidOHLC"].iloc[0])
        self.assertFalse(result["Tradable"].iloc[0])




if __name__ == "__main__":
    unittest.main()
