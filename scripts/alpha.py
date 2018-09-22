import pandas as pd
import requests

OPEN_COL = '1. open'
CLOSE_COL = '2. high'
HIGH_COL = '3. low'
LOW_COL = '4. close'
ADJUSTED_CLOSE_COL = '5. adjusted close'
VOLUME_COL = '6. volume'
DIVIDENT_AMOUNT_COL = '7. dividend amount'
SPLIT_COEFFICIENT_COL = '8. split coefficient'
LABEL_COL = 'Label'
FORECAST_COL = 'Forecast'
DAILY_PCT_CHANGE_COL = 'Daily pct change'
HL_PCT_CHANGE_COL = 'H/L pct change'


class AlphaVantage:
    API_URL = 'https://www.alphavantage.co/query'
    API_KEY = 'yM2zzAs6_DxdeT86rtZY'
    DAILY_ADJUSTED = 'TIME_SERIES_DAILY_ADJUSTED'
    FULL = 'full'

    def daily_adjusted_raw(self, ticker):
        data = {"apikey": self.API_KEY,
                "symbol": ticker,
                "function": self.DAILY_ADJUSTED,
                "outputsize": self.FULL
                }
        r = requests.get(self.API_URL, params=data)
        return r.json()

    def daily_adjusted(self, ticker):
        json = self.daily_adjusted_raw(ticker)
        keys = list(json.keys())
        series = keys[1]
        df = pd.DataFrame.from_dict(json[series], orient='index')
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        return df
