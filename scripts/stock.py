import requests
import pandas as pd


class AlphaVantage:
    API_URL = 'https://www.alphavantage.co/query'
    API_KEY = 'yM2zzAs6_DxdeT86rtZY'
    DAILY_ADJUSTED = 'TIME_SERIES_DAILY_ADJUSTED'

    def daily_adjusted_raw(self, ticker):
        data = {"apikey": self.API_KEY,
                "symbol": ticker,
                "function": self.DAILY_ADJUSTED
                }
        r = requests.get(self.API_URL, params=data)
        return r.json()
