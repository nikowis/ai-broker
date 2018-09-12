import requests


class AlphaVantage:
    API_URL = 'https://www.alphavantage.co/query?function='
    '&symbol=GOOGLyM2zzAs6_DxdeT86rtZY'
    API_KEY = '&apikey=yM2zzAs6_DxdeT86rtZY'
    DAILY_ADJUSTED = 'TIME_SERIES_DAILY_ADJUSTED'
    SYMBOL = 'symbol='
    OUTPUT = '&outputsize=full'

    def daily_adjusted_raw(self, ticker):
        url = self.API_URL + self.DAILY_ADJUSTED + '&' + self.SYMBOL + ticker + self.OUTPUT + self.API_KEY
        r = requests.get(url)
        return r.json()
