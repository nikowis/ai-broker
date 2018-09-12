import unittest
import scripts.stock as stck


class AlphaVantageTests(unittest.TestCase):

    def test_daily_adjusted_raw(self):
        s = stck.AlphaVantage()
        result = s.daily_adjusted_raw('GOOGL')
        self.assertEqual(len(result), 2)

    def test_daily_adjusted(self):
        s = stck.AlphaVantage()
        result = s.daily_adjusted('GOOGL')
        self.assertTrue(result.size > 1000)

    def test_columns(self):
        s = stck.AlphaVantage()
        result = s.daily_adjusted('GOOGL')
        self.assertTrue(result[stck.AlphaVantage.OPEN_COL].size > 1000)
        self.assertTrue(result[stck.AlphaVantage.CLOSE_COL].size > 1000)
        self.assertTrue(result[stck.AlphaVantage.HIGH].size > 1000)
        self.assertTrue(result[stck.AlphaVantage.LOW_COL].size > 1000)
        self.assertTrue(result[stck.AlphaVantage.ADJUSTED_CLOSE_COL].size > 1000)
        self.assertTrue(result[stck.AlphaVantage.VOLUME_COL].size > 1000)
        self.assertTrue(result[stck.AlphaVantage.DIVIDENT_AMOUT].size > 1000)
        self.assertTrue(result[stck.AlphaVantage.SPLIT_COEFFICIENT].size > 1000)
        self.assertTrue(result[stck.AlphaVantage.OPEN_COL].size > 1000)
        with self.assertRaises(KeyError):
            var = result["randomcolssa32"]


if __name__ == '__main__':
    unittest.main()
