import numpy as np
import pandas as pd

from db import db_access
import db.stock_constants as const


def process_data():
    db = db_access.create_db_connection()
    stock_collection_raw = db_access.stock_collection(db, False)
    stock_processed_collection = db_access.stock_collection(db, True)

    for stock in stock_collection_raw.find():
        symbol = stock[const.SYMBOL]
        print(symbol)
        if len(stock) > 3000 and stock_processed_collection.count({const.SYMBOL: symbol}) < 1:
            df = prepare_df(stock)
            processed_dict = df.to_dict(const.INDEX)
            processed_dict[const.SYMBOL] = symbol
            stock_processed_collection.insert(processed_dict)


def prepare_df(stock):
    stock.pop(const.ID, None)
    stock.pop(const.SYMBOL, None)
    df = pd.DataFrame.from_dict(stock, orient=const.INDEX)
    df = df.astype(float)
    # df.index = pd.to_datetime(df.index)
    df[const.LABEL_COL] = df[const.ADJUSTED_CLOSE_COL].shift(-const.FORECAST_DAYS)
    df[const.DAILY_PCT_CHANGE_COL] = (df[const.LABEL_COL] - df[const.ADJUSTED_CLOSE_COL]) / df[
        const.ADJUSTED_CLOSE_COL] * 100.0
    df[const.LABEL_DISCRETE_COL] = df[const.DAILY_PCT_CHANGE_COL].apply(
        lambda pct: np.NaN if pd.isna(pct)
        else const.FALL_VALUE if pct < -const.TRESHOLD else const.RISE_VALUE if pct > const.TRESHOLD else const.IDLE_VALUE)
    df[const.LABEL_BINARY_COL] = df[const.DAILY_PCT_CHANGE_COL].apply(
        lambda pct: np.NaN if pd.isna(pct)
        else const.FALL_VALUE if pct < 0 else const.IDLE_VALUE if pct >= 0 else const.RISE_VALUE)

    return df


if __name__ == "__main__":
    process_data()
