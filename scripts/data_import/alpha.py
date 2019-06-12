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

    def __init__(self, key) -> None:
        super().__init__()
        self.key = key

    FULL = 'full'
    COMPACT = 'compact'

    def data_raw(self, ticker, data_type=DataType.DAILY_ADJUSTED):
        data = {"apikey": self.key,
                "symbol": ticker,
                "function": data_type,
                "outputsize": self.FULL
                }
        if data_type == AlphaVantage.DataType.INTRADAY:
            data['interval'] = '1min'

        return requests.get(self.API_URL, params=data)

    def technical_indicator(self, ticker, indicator, time_period=None, interval='daily', series_type='close'):
        data = {"apikey": self.key,
                "symbol": ticker,
                "interval": interval,
                "function": indicator,
                "series_type": series_type
                }
        if time_period is not None:
            data['time_period'] = time_period

        return requests.get(self.API_URL, params=data)
