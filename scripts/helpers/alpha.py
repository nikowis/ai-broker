import json
import os

import pandas as pd
import requests

OPEN_COL = '1. open'
HIGH_COL = '2. high'
LOW_COL = '3. low'
CLOSE_COL = '4. close'
ADJUSTED_CLOSE_COL = '5. adjusted close'
VOLUME_COL = '6. volume'
VOLUME_INTRADAY_COL = '5. volume'
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
    class DataType:
        DAILY_ADJUSTED = 'TIME_SERIES_DAILY_ADJUSTED'
        INTRADAY = 'TIME_SERIES_INTRADAY'

    API_URL = 'https://www.alphavantage.co/query'
    API_KEY = 'yM2zzAs6_DxdeT86rtZY'
    #TX1OLY36K73S9MS9
    #I7RUE3LA4PSXDJU6
    #ULDORYWPDU2S2E6X

    FULL = 'full'
    COMPACT = 'compact'
    API_CACHE_PATH = './../target/api_cache/'

    def data_raw(self, ticker, data_type=DataType.DAILY_ADJUSTED):
        data = {"apikey": self.API_KEY,
                "symbol": ticker,
                "function": data_type,
                "outputsize": self.FULL
                }
        if data_type == AlphaVantage.DataType.INTRADAY:
            data['interval'] = '1min'

        return requests.get(self.API_URL, params=data)

    def data(self, ticker, data_type=DataType.DAILY_ADJUSTED, cache=True):
        cache_file_path = self.API_CACHE_PATH + ticker + '_' + data_type + '.json'
        if not cache or not os.path.exists(cache_file_path):
            if not os.path.exists(self.API_CACHE_PATH):
                os.makedirs(self.API_CACHE_PATH)
            response = self.data_raw(ticker, data_type).json()
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
