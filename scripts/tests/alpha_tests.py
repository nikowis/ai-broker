import unittest
import alpha as stck


class AlphaVantageTests(unittest.TestCase):

    def test_daily_adjusted_raw(self):
        s = stck.AlphaVantage()
        result = s.data_raw('GOOGL', s.DataType.DAILY_ADJUSTED).json()
        self.assertEqual(len(result), 2)

    def test_daily_adjusted(self):
        s = stck.AlphaVantage()
        result = s.data('GOOGL')
        self.assertTrue(result.size > 1000)

    def test_columns(self):
        s = stck.AlphaVantage()
        result = s.data('GOOGL')
        self.assertTrue(result[stck.OPEN_COL].size > 1000)
        self.assertTrue(result[stck.CLOSE_COL].size > 1000)
        self.assertTrue(result[stck.HIGH_COL].size > 1000)
        self.assertTrue(result[stck.LOW_COL].size > 1000)
        self.assertTrue(result[stck.ADJUSTED_CLOSE_COL].size > 1000)
        self.assertTrue(result[stck.VOLUME_COL].size > 1000)
        self.assertTrue(result[stck.DIVIDENT_AMOUNT_COL].size > 1000)
        self.assertTrue(result[stck.SPLIT_COEFFICIENT_COL].size > 1000)
        self.assertTrue(result[stck.OPEN_COL].size > 1000)
        with self.assertRaises(KeyError):
            var = result["randomcolssa32"]


if __name__ == '__main__':
    unittest.main()
