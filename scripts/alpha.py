import pandas as pd
import requests
import os
import json

OPEN_COL = '1. open'
CLOSE_COL = '2. high'
HIGH_COL = '3. low'
LOW_COL = '4. close'
ADJUSTED_CLOSE_COL = '5. adjusted close'
VOLUME_COL = '6. volume'
DIVIDENT_AMOUNT_COL = '7. dividend amount'
SPLIT_COEFFICIENT_COL = '8. split coefficient'
LABEL_COL = 'Label'
LABEL_DISCRETE_COL = 'Label discrete'
FORECAST_FOR_TODAY_COL = 'Forecast for today'
FORECAST_FUTURE_COL = 'Forecast future'
FORECAST_PCT_CHANGE_COL = 'Forecast pct change'
FORECAST_DISCRETE_COL = 'Forecast discrete'
DAILY_PCT_CHANGE_COL = 'Daily pct change'
HL_PCT_CHANGE_COL = 'H/L pct change'


class AlphaVantage:
    API_URL = 'https://www.alphavantage.co/query'
    API_KEY = 'yM2zzAs6_DxdeT86rtZY'
    DAILY_ADJUSTED = 'TIME_SERIES_DAILY_ADJUSTED'
    FULL = 'full'
    API_CACHE_PATH = './../target/api_cache/'

    def daily_adjusted_raw(self, ticker):
        data = {"apikey": self.API_KEY,
                "symbol": ticker,
                "function": self.DAILY_ADJUSTED,
                "outputsize": self.FULL
                }
        r = requests.get(self.API_URL, params=data)
        return r.json()

    def daily_adjusted(self, ticker, cache=True):
        cache_file_path = self.API_CACHE_PATH + ticker + '_' + self.DAILY_ADJUSTED + '.json'
        if not cache or not os.path.exists(cache_file_path):
            if not os.path.exists(self.API_CACHE_PATH):
                os.makedirs(self.API_CACHE_PATH)
            response = self.daily_adjusted_raw(ticker)
            with open(cache_file_path, "w") as text_file:
                text_file.write(json.dumps(response))
        else:
            with open(cache_file_path, "r") as text_file:
                response = json.loads(text_file.read())
                
        keys = list(response.keys())
        series = keys[1]
        df = pd.DataFrame.from_dict(response[series], orient='index')
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        return df
