import unittest
import scripts.stock as stck


class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        s = stck.AlphaVantage()
        result = s.daily_adjusted_raw('GOOGL')
        print(result)
        self.assertTrue(len(result) == 2)


if __name__ == '__main__':
    unittest.main()
