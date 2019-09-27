import os

import numpy as np
import pandas as pd

import stock_constants as const
from statistics import mean

DIR = './../predictions/'
DIR_COMPANIES = DIR + 'companies/'


def save_to_company_csv():
    df_companies = {}
    for c in const.BASE_COMPANIES:
        df_companies[c] = pd.DataFrame()

    for filename in os.listdir(DIR):
        if filename.endswith(".csv"):
            print(filename)
            df = pd.read_csv(DIR + filename)
            for index, row in df.iterrows():
                print(row["date"], row["symbol"])
                company_df = df_companies[row["symbol"]]
                df_companies[row["symbol"]] = company_df.append(row, ignore_index=True)

    for c in const.BASE_COMPANIES:
        print(df_companies[c].head())
        df = df_companies[c]
        df.index = pd.to_datetime(df.index)
        df[const.LABEL_COL] = df['close price'].shift(-const.FORECAST_DAYS)
        df[const.DAILY_PCT_CHANGE_COL] = (df[const.LABEL_COL] - df['close price']) / df[
            'close price'] * 100.0
        df[const.LABEL_DISCRETE_COL] = df[const.DAILY_PCT_CHANGE_COL].apply(
            lambda pct: np.NaN if pd.isna(pct)
            else const.FALL_VALUE if pct < -const.TRESHOLD else const.RISE_VALUE if pct > const.TRESHOLD else const.IDLE_VALUE)
        df[const.LABEL_BINARY_COL] = df[const.DAILY_PCT_CHANGE_COL].apply(
            lambda pct: np.NaN if pd.isna(pct)
            else const.FALL_VALUE if pct < 0 else const.RISE_VALUE if pct >= 0 else const.IDLE_VALUE)
        df_companies[c] = df
        df_companies[c].to_csv('{0}/predictions-{1}.csv'.format(DIR_COMPANIES, c), index=False)


def analyze_company(company):
    df = pd.read_csv(DIR_COMPANIES + 'predictions-' + company + '.csv', index_col=2)
    df.dropna(inplace=True)
    binary_count = 0
    discrete_count = 0
    rows = 0
    # print(df.tail())
    for index, row in df.iterrows():
        rows = rows + 1
        if row["binary prediction value"] == row["Label binary"]:
            binary_count = binary_count + 1
        if row["discrete prediction value"] == row["Label discrete"]:
            discrete_count = discrete_count + 1

    binary_acc = binary_count / rows
    discrete_acc = discrete_count / rows
    print('{0}: binary {1}   discrete {2}'.format(company, binary_acc, discrete_acc))
    return binary_acc, discrete_acc


if __name__ == "__main__":
    # save_to_company_csv()
    bins = []
    discretes = []
    for c in const.BASE_COMPANIES:
        binary_acc, discrete_acc = analyze_company(c)
        bins.append(binary_acc)
        discretes.append(discrete_acc)

    print('Binary avg {0}   discrete avg {1}'.format(mean(bins), mean(discretes)))

    print("Program finished")
