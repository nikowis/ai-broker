import os
import time

import numpy as np
import pandas as pd

import stock_constants as const
from data_import import alpha
from data_import.db_access import create_db_connection, stock_collection

TI_BBANDS = 'BBANDS'
TI_RSI = 'RSI'
TI_STOCH = 'STOCH'
TI_MACD = 'MACD'
TI_SMA = 'SMA'
TI_ROC = 'ROC'
TI_TR = 'TRANGE'
TI_MOM = 'MOM'
TI_WILLR = 'WILLR'  # Williams' %R
TI_APO = 'APO'  # absolute price oscillator
TI_ADX = 'ADX'  # absolute price oscillator
TI_CCI = 'CCI'  # absolute price oscillator
TI_AD = 'AD'  # absolute price oscillator

SYMBOL_KEY = "symbol"

API_MAX_PER_MINUTE_CALLS = 5
API_MAX_DAILY = 400

SELECTED_SYM = 'GOOGL'

API_KEYS = ['ULDORYWPDU2S2E6X', 'yM2zzAs6_DxdeT86rtZY', 'TX1OLY36K73S9MS9', 'I7RUE3LA4PSXDJU6', '41KVI2PCCMZ09Y69']


class Importer:

    def __init__(self) -> None:
        super().__init__()
        self.minute_count = 0
        self.daily_count = 0
        self.api_key_index = 0
        self.db = create_db_connection()
        self.api = alpha.AlphaVantage(API_KEYS[self.api_key_index])

    def json_to_df(self, json):
        json.pop(const.ID, None)
        json.pop(const.SYMBOL_KEY, None)
        df = pd.DataFrame.from_dict(json, orient=const.INDEX)
        df = df.astype(float)
        return df

    def df_to_json(self, df, ticker):
        json = df.to_dict(const.INDEX)
        json[const.SYMBOL_KEY] = ticker
        return json

    def import_one(self, sym):
        if stock_collection(self.db, False).count({SYMBOL_KEY: sym}) > 0:
            print('Found object with symbol ', sym)
        else:
            print('Didnt find object with symbol ', sym)
            raw_json = self.api.data_raw(sym).json(object_pairs_hook=self.remove_dots)
            keys = list(raw_json.keys())
            if len(keys) < 2:
                print('Symbol ', sym, 'not existing in alpha vantage')
                print(str(raw_json))
                return
            time_series_key = keys[1]
            time_series = raw_json[time_series_key]
            time_series[SYMBOL_KEY] = sym
            stock_collection(self.db, False).insert(time_series)
            self.increment_counters_sleep()

    def increment_counters_sleep(self):
        self.minute_count = self.minute_count + 1
        self.daily_count = self.daily_count + 1
        if self.daily_count >= API_MAX_DAILY:
            self.minute_count = 0
            self.daily_count = 0
            self.api_key_index = self.api_key_index + 1
            self.api = alpha.AlphaVantage(API_KEYS[self.api_key_index])
            print('####################CHANGING API KEY##################')
            time.sleep(10)
        if self.minute_count >= API_MAX_PER_MINUTE_CALLS:
            print('Sleeping.')
            time.sleep(65)
            self.minute_count = 0

    def remove_dots(self, items):
        result = {}
        for key, value in items:
            key = key.replace('.', ' ')
            result[key] = value
        return result

    def import_all(self, symbols):
        for sym in symbols:
            self.import_one(sym)

    def import_all_technical_indicators(self, tickers):
        for ticker in tickers:
            json = stock_collection(self.db, False).find_one({const.SYMBOL_KEY: ticker})
            df = self.json_to_df(json)
            # https://journals.plos.org/plosone/article/figure?id=10.1371/journal.pone.0122385.t001
            self.import_technical_indicator(ticker, df, TI_SMA, const.SMA_5_COL, time_period=5)
            self.import_technical_indicator(ticker, df, TI_SMA, const.SMA_10_COL, time_period=10)
            self.import_technical_indicator(ticker, df, TI_SMA, const.SMA_20_COL, time_period=20)
            self.import_technical_indicator(ticker, df, TI_ROC, const.ROC_5_COL, time_period=5)
            self.import_technical_indicator(ticker, df, TI_ROC, const.ROC_10_COL, time_period=10)
            self.import_technical_indicator(ticker, df, TI_TR, const.TR_COL)
            self.import_technical_indicator(ticker, df, TI_MOM, const.MOM_5_COL, time_period=5)
            self.import_technical_indicator(ticker, df, TI_MOM, const.MOM_10_COL, time_period=10)
            self.import_technical_indicator(ticker, df, TI_MACD, const.MACD_COL)
            self.import_technical_indicator(ticker, df, TI_STOCH, const.STOCH_K_COL)
            self.import_technical_indicator(ticker, df, TI_WILLR, const.WILLR_5_COL, time_period=5)
            self.import_technical_indicator(ticker, df, TI_WILLR, const.WILLR_10_COL, time_period=10)
            self.import_technical_indicator(ticker, df, TI_APO, const.APO_5_COL, time_period=5)  # OSCILIATOR
            self.import_technical_indicator(ticker, df, TI_APO, const.APO_10_COL, time_period=10)  # OSCILIATOR
            self.import_technical_indicator(ticker, df, TI_RSI, const.RSI_5_COL, time_period=5)
            self.import_technical_indicator(ticker, df, TI_RSI, const.RSI_10_COL, time_period=10)
            self.import_technical_indicator(ticker, df, TI_ADX, const.ADX_5_COL, time_period=5)
            self.import_technical_indicator(ticker, df, TI_ADX, const.ADX_10_COL, time_period=10)
            self.import_technical_indicator(ticker, df, TI_CCI, const.CCI_5_COL, time_period=5)
            self.import_technical_indicator(ticker, df, TI_CCI, const.CCI_10_COL, time_period=10)
            self.import_technical_indicator(ticker, df, TI_AD, const.AD_COL)
            self.import_technical_indicator(ticker, df, TI_BBANDS, const.BBANDS_10_RLB_COL, time_period=10)
            self.import_technical_indicator(ticker, df, TI_BBANDS, const.BBANDS_20_RLB_COL, time_period=20)

    def import_technical_indicator(self, ticker, df, indicator, col_name, time_period=None):
        if col_name not in df.columns:
            print('Importing ', indicator, ' for ', ticker)
            raw_json = self.api.technical_indicator(ticker, indicator, time_period=time_period).json(
                object_pairs_hook=self.remove_dots)
            keys = list(raw_json.keys())
            if len(keys) < 2:
                print('Symbol ', ticker, 'not existing in alpha vantage')
                print(str(raw_json))
                return
            time_series_key = keys[1]
            time_series = raw_json[time_series_key]
            indicator_df = self.json_to_df(time_series)
            if 'MACD' == indicator:
                prefix = col_name + ' '
                df[col_name] = indicator_df[indicator]
                df[prefix + 'Hist'] = indicator_df['MACD_Hist']
                df[prefix + 'Signal'] = indicator_df['MACD_Signal']
            elif 'STOCH' == indicator:
                prefix = indicator + ' '
                df[prefix + 'SlowK'] = indicator_df['SlowK']
                df[prefix + 'SlowD'] = indicator_df['SlowD']
            elif 'BBANDS' == indicator:
                prefix = str(time_period) + '-' + indicator + ' '
                df[prefix + 'Real Lower Band'] = indicator_df['Real Lower Band']
                df[prefix + 'Real Upper Band'] = indicator_df['Real Upper Band']
                df[prefix + 'Real Middle Band'] = indicator_df['Real Middle Band']
            elif 'AD' == indicator:
                df[col_name] = indicator_df['Chaikin A/D']
            else:
                df[col_name] = indicator_df[indicator]
            processed_json = self.df_to_json(df, ticker)
            stock_collection(self.db, False).remove({const.SYMBOL_KEY: ticker})
            stock_collection(self.db, False).insert(processed_json)
            self.increment_counters_sleep()

    def process_data(self):
        stock_collection_raw = stock_collection(self.db, False)
        stock_processed_collection = stock_collection(self.db, True)

        for stock in stock_collection_raw.find():
            symbol = stock[const.SYMBOL_KEY]
            if stock_collection(self.db, True).count({SYMBOL_KEY: symbol}) > 0:
                print('Not processing ', symbol, ' - already processed')
            else:
                df = self.json_to_df(stock)
                df[const.LABEL_COL] = df[const.ADJUSTED_CLOSE_COL].shift(-const.FORECAST_DAYS)
                df[const.DAILY_PCT_CHANGE_COL] = (df[const.LABEL_COL] - df[const.ADJUSTED_CLOSE_COL]) / df[
                    const.ADJUSTED_CLOSE_COL] * 100.0
                df[const.LABEL_DISCRETE_COL] = df[const.DAILY_PCT_CHANGE_COL].apply(
                    lambda pct: np.NaN if pd.isna(pct)
                    else const.FALL_VALUE if pct < -const.TRESHOLD else const.RISE_VALUE if pct > const.TRESHOLD else const.IDLE_VALUE)
                df[const.LABEL_BINARY_COL] = df[const.DAILY_PCT_CHANGE_COL].apply(
                    lambda pct: np.NaN if pd.isna(pct)
                    else const.FALL_VALUE if pct < 0 else const.RISE_VALUE if pct >= 0 else const.IDLE_VALUE)
                df[const.HL_PCT_CHANGE_COL] = (df[const.HIGH_COL] - df[const.LOW_COL]) / df[
                    const.HIGH_COL] * 100
                df[const.SMA_DIFF_COL] = df[const.SMA_10_COL] - df[const.SMA_5_COL]
                df[const.SMA_DIFF2_COL] = df[const.SMA_20_COL] - df[const.SMA_5_COL]
                df[const.ROC_DIFF_COL] = df[const.ROC_10_COL] - df[const.ROC_5_COL]
                df[const.MOM_DIFF_COL] = df[const.MOM_10_COL] - df[const.MOM_5_COL]
                df[const.WILLR_DIFF_COL] = df[const.WILLR_10_COL] - df[const.WILLR_5_COL]
                df[const.APO_DIFF_COL] = df[const.APO_10_COL] - df[const.APO_5_COL]
                df[const.RSI_DIFF_COL] = df[const.RSI_10_COL] - df[const.RSI_5_COL]
                df[const.ADX_DIFF_COL] = df[const.ADX_10_COL] - df[const.ADX_5_COL]
                df[const.CCI_DIFF_COL] = df[const.CCI_10_COL] - df[const.CCI_5_COL]
                df[const.STOCH_D_DIFF_COL] = df[const.STOCH_D_COL] - df[const.STOCH_D_COL].shift(-1)
                df[const.STOCH_K_DIFF_COL] = df[const.STOCH_K_COL] - df[const.STOCH_K_COL].shift(-1)
                df[const.DISPARITY_5_COL] = 100 * df[const.ADJUSTED_CLOSE_COL] / df[const.SMA_5_COL]
                df[const.DISPARITY_10_COL] = 100 * df[const.ADJUSTED_CLOSE_COL] / df[const.SMA_10_COL]
                df[const.DISPARITY_20_COL] = 100 * df[const.ADJUSTED_CLOSE_COL] / df[const.SMA_20_COL]
                df[const.BBANDS_10_DIFF_COL] = df[const.BBANDS_10_RUB_COL] - df[const.BBANDS_10_RLB_COL]
                df[const.BBANDS_20_DIFF_COL] = df[const.BBANDS_20_RUB_COL] - df[const.BBANDS_20_RLB_COL]
                df[const.PRICE_BBANDS_LOW_10_COL] = (df[const.ADJUSTED_CLOSE_COL] - df[const.BBANDS_10_RLB_COL]) / df[
                    const.BBANDS_10_RLB_COL]
                df[const.PRICE_BBANDS_LOW_20_COL] = (df[const.ADJUSTED_CLOSE_COL] - df[const.BBANDS_20_RLB_COL]) / df[
                    const.BBANDS_20_RLB_COL]
                df[const.PRICE_BBANDS_UP_10_COL] = (df[const.ADJUSTED_CLOSE_COL] - df[const.BBANDS_10_RUB_COL]) / df[
                    const.BBANDS_10_RUB_COL]
                df[const.PRICE_BBANDS_UP_20_COL] = (df[const.ADJUSTED_CLOSE_COL] - df[const.BBANDS_20_RUB_COL]) / df[
                    const.BBANDS_20_RUB_COL]

                processed_dict = self.df_to_json(df, symbol)
                stock_processed_collection.insert(processed_dict)
                print('Processed ', symbol)

    def export_to_csv_files(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        stock_processed_collection = stock_collection(self.db, True)

        for stock in stock_processed_collection.find():
            symbol = stock[const.SYMBOL_KEY]
            df = self.json_to_df(stock)
            file = path + '/' + symbol + '.csv'
            df.to_csv(file, encoding='utf-8')

            print('Exported to csv ', symbol)


if __name__ == "__main__":
    imp = Importer()
    imp.import_all(['SIRI', 'MYL', 'SYMC', 'KHC', 'JD', 'AMD', 'FAST', 'AAL', 'MU', 'CTRP'])
    imp.import_all_technical_indicators(['SIRI', 'MYL', 'SYMC', 'KHC', 'JD', 'AMD', 'FAST', 'AAL', 'MU', 'CTRP'])
    imp.process_data()
    imp.export_to_csv_files('./../../target/data')
    # dflist, _ = csv_importer.import_data_from_files([SELECTED_SYM], './../../target/data')

    print("Importing finished")
