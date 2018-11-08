import numpy as np
import pandas as pd

from db import db_access

SYMBOL = 'symbol'

ID = '_id'

OPEN_COL = '1  open'
HIGH_COL = '2  high'
LOW_COL = '3  low'
CLOSE_COL = '4  close'
ADJUSTED_CLOSE_COL = '5  adjusted close'
VOLUME_COL = '6  volume'
VOLUME_INTRADAY_COL = '5  volume'
DIVIDENT_AMOUNT_COL = '7  dividend amount'
SPLIT_COEFFICIENT_COL = '8  split coefficient'
LABEL_COL = 'Label'
FORECAST_DAYS = 1
LABEL_DISCRETE_COL = 'Label discrete'
FORECAST_FOR_TODAY_COL = 'Forecast for today'
FORECAST_FUTURE_COL = 'Forecast future'
FORECAST_PCT_CHANGE_COL = 'Forecast pct change'
FORECAST_DISCRETE_COL = 'Forecast discrete'
DAILY_PCT_CHANGE_COL = 'Daily pct change'
HL_PCT_CHANGE_COL = 'H/L pct change'
FALL_VALUE = -1
IDLE_VALUE = 0
RISE_VALUE = 1
TRESHOLD = 0.4


def process_data():
    db = db_access.create_db_connection()
    stock_collection_raw = db_access.stock_collection(db)
    stock_processed_collection = db_access.stock_collection(db, True)

    for stock in stock_collection_raw.find():
        symbol = stock[SYMBOL]
        print(symbol)
        if len(stock) > 3000 and stock_processed_collection.count({SYMBOL: symbol}) < 1:
            df = prepare_df(stock)
            processed_dict = df.to_dict('index')
            processed_dict[SYMBOL] = symbol
            stock_processed_collection.insert(processed_dict)


def prepare_df(stock):
    stock.pop(ID, None)
    stock.pop(SYMBOL, None)
    df = pd.DataFrame.from_dict(stock, orient='index')
    df = df.astype(float)
    #df.index = pd.to_datetime(df.index)
    df[LABEL_COL] = df[ADJUSTED_CLOSE_COL].shift(-FORECAST_DAYS)
    df[DAILY_PCT_CHANGE_COL] = (df[LABEL_COL] - df[ADJUSTED_CLOSE_COL]) / df[ADJUSTED_CLOSE_COL] * 100.0
    df[LABEL_DISCRETE_COL] = df[DAILY_PCT_CHANGE_COL].apply(
        lambda pct: np.NaN if pd.isna(
            pct) else FALL_VALUE if pct < -TRESHOLD else RISE_VALUE if pct > TRESHOLD else IDLE_VALUE)
    return df


if __name__ == "__main__":
    process_data()
