import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style

import csv_importer
import stock_constants

ROC_AUC_COL = 'roc_auc'
TRAIN_TIME_COL = 'train_time'
ACCURACY_COL = 'accuracy'
RESULT_PATH = './../../target/results/'
IMG_PATH = './../../target/results/simulation_imgs/'

if not os.path.exists(IMG_PATH):
    os.makedirs(IMG_PATH)


def analyze_csv(filepath, examined_params, print_latex=False):
    df = pd.read_csv(RESULT_PATH + filepath)

    print('Read {0}'.format(filepath))

    split_params = examined_params.split(',')
    for examined_param in split_params:
        df[examined_param] = df[examined_param].astype(str)

    mean_groupby = df.drop('ID', axis=1).groupby(split_params).mean()
    print('Examination of {0}:\n {1}'.format(examined_params, mean_groupby))
    if print_latex:
        print(mean_groupby.to_latex())
    return mean_groupby


def analyze_nn_layers(filepath, examined_param='layers', print_latex=True):
    mean_groupby = analyze_csv(filepath, examined_param, False)
    if print_latex:
        only_roc_df = mean_groupby.drop([TRAIN_TIME_COL, ACCURACY_COL], axis=1).round(3)
        one_layer_df = only_roc_df.filter(regex='\[\s*\d*\s*\]', axis=0).T
        two_layer_df = only_roc_df.filter(regex='\[\s*\d+\s*,\s*\d+\s*\]', axis=0).T
        three_layer_df = only_roc_df.filter(regex='\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]', axis=0).T

        print(one_layer_df.to_latex())
        print(two_layer_df.to_latex())
        print(three_layer_df.to_latex())


def analyze_final(filepath):
    df = pd.read_csv(RESULT_PATH + filepath)

    print('Read {0}'.format(filepath))

    mean_groupby = df.drop(['ID', 'train_time'], axis=1).groupby('ticker').mean()
    avg_groupby = df.drop(['ID', 'train_time', 'ticker'], axis=1).mean()
    print('Final analyzis:\n {0}'.format(mean_groupby))
    print('Average:\n {0}'.format(avg_groupby))

    print(mean_groupby.to_latex())
    return mean_groupby


def analyze_simulation(filepath, print_buy_and_hold=False):
    df = pd.read_csv(RESULT_PATH + filepath)

    df['buy_and_hold_balance'] = df['buy_and_hold_balance'] / df['budget'] * 100
    df['balance'] = df['balance'] / df['budget'] * 100
    df = df.sort_values(by='ticker')

    if print_buy_and_hold:
        print(df.round(2).to_latex(index=False, columns=['ticker', 'buy_and_hold_balance']))

    mean_df = df.drop(['ticker', 'budget'], axis=1).mean().round(2)
    print('Simulation {0} analyzis: avg balance {1}, avg buy and hold {2}'.format(filepath, mean_df['balance'],
                                                                                  mean_df['buy_and_hold_balance']))
    print(df.round(2).to_latex(index=False, columns=['ticker', 'balance']))


def analyze_simulation_details(filepath, symbol, start_date, print_buy_and_hold=False):
    df_list, _ = csv_importer.import_data_from_files([symbol], './../../target/data/')
    df = df_list[0]
    df = df[(df.index > start_date)]

    trade_df = pd.read_csv(RESULT_PATH + filepath)
    trade_df.set_index('date', inplace=True)
    trade_df.index = pd.to_datetime(trade_df.index)

    style.use('ggplot')
    df[stock_constants.ADJUSTED_CLOSE_COL].plot(kind='line', x_compat=True, label='Cena zamknięcia', color='black')
    plt.title(symbol)
    plt.ylabel('Cena zamknięcia (USD)')
    plt.xlabel('Data')

    index_buy = None
    index_sell = None
    reds = 0
    greens = 0
    for index, row in trade_df.iterrows():
        if row['buy'] == 1:
            index_buy = index
            if index_buy is not None and index_sell is not None:
                plt.axvspan(index_sell, index_buy, alpha=0.4, color='red', label="_" * reds + "Gotówka w portfelu")
                reds = reds + 1
        elif row['sell'] == 1:
            index_sell = index
            if index_buy is not None and index_sell is not None:
                plt.axvspan(index_buy, index_sell, alpha=0.4, color='green', label="_" * greens + "Akcje w portfelu")
                greens = greens + 1

    plt.legend()

    plt.savefig('{}/{}-buy_and_sell_plot.png'.format(IMG_PATH, symbol))
    plt.savefig('{}/{}-buy_and_sell_plot.pdf'.format(IMG_PATH, symbol), format='pdf', dpi=1000)
    # plt.show()
    plt.close()

    style.use('ggplot')
    trade_df['balance'] = trade_df['balance'] / 1000
    trade_df['buy_and_hold_balance'] = trade_df['buy_and_hold_balance'] / 1000
    trade_df['balance'].plot(kind='line', x_compat=True, label='Wartość portfela klasyfikatora')
    trade_df['buy_and_hold_balance'].plot(kind='line', x_compat=True, label='Wartość portfela buy and hold')
    plt.title(symbol)
    plt.ylabel('Wartość portfela (tys. USD)')
    plt.xlabel('Data')
    plt.legend()
    plt.savefig('{}/{}-balance-plot.png'.format(IMG_PATH, symbol))
    plt.savefig('{}/{}-balance-plot.pdf'.format(IMG_PATH, symbol), format='pdf', dpi=1000)
    # plt.show()
    plt.close()


def average_trades(dirpaths):
    for dirpath in dirpaths:
        transaction_count = []
        for filename in os.listdir(RESULT_PATH + dirpath):
            if filename.endswith(".csv"):
                trade_df = pd.read_csv(RESULT_PATH + dirpath + '/' + filename)
                trade_df.set_index('date', inplace=True)
                trade_df.index = pd.to_datetime(trade_df.index)
                transaction_count.append(len(trade_df))
        print('{0} average {1} transactions'.format(dirpath, np.mean(transaction_count)))


if __name__ == '__main__':
    # analyze_csv('results-nn-pca-GOOGL-binary.csv', 'pca', True)
    # analyze_csv('results-nn-pca-GOOGL-discrete.csv', 'pca', True)
    # analyze_nn_layers('results-nn-layers-GOOGL-binary.csv')
    # analyze_nn_layers('results-nn-layers-GOOGL-discrete.csv')
    # analyze_csv('results-nn-max_train_window_size-GOOGL-discrete.csv', 'max_train_window_size', True)
    # analyze_csv('results-nn-walk_forward_test_window_size-GOOGL-binary.csv', 'walk_forward_test_window_size', True)
    # analyze_csv('results-nn-walk_forward_test_window_size-GOOGL-discrete.csv', 'walk_forward_test_window_size', True)
    # analyze_final("results-nn-final-binary.csv")
    # analyze_final("results-nn-final-discrete.csv")
    # analyze_csv('results-svm-pca-GOOGL-binary.csv', 'pca', True)
    # analyze_csv('results-svm-pca-GOOGL-discrete.csv', 'pca', True)
    # analyze_csv('results-svm-kernel-GOOGL-binary.csv', 'kernel', True)
    # analyze_csv('results-svm-kernel-GOOGL-discrete.csv', 'kernel', True)
    # analyze_csv('results-svm-c-GOOGL-binary.csv', 'c', True)
    # analyze_csv('results-svm-c-GOOGL-discrete.csv', 'c', True)
    # analyze_csv('results-svm-c-kernel-GOOGL-binary.csv', 'c', True)
    # analyze_csv('results-svm-c-kernel-GOOGL-discrete.csv', 'c', True)
    # analyze_csv('results-svm-walk_forward_test_window_size-GOOGL-binary.csv', 'walk_forward_test_window_size', True)
    # analyze_csv('results-svm-walk_forward_test_window_size-GOOGL-discrete.csv', 'walk_forward_test_window_size', True)
    # analyze_final('results-svm-final-binary.csv')
    # analyze_final('results-svm-final-discrete.csv')
    # analyze_csv('results-lgbm-pca-GOOGL-binary.csv', 'pca', True)
    # analyze_csv('results-lgbm-pca-GOOGL-discrete.csv', 'pca', True)
    # analyze_csv('results-lgbm-feature_fraction-GOOGL-discrete.csv', 'feature_fraction', True)
    # analyze_csv('results-lgbm-feature_fraction-GOOGL-binary.csv', 'feature_fraction', True)
    # analyze_csv('results-lgbm-boosting-GOOGL-binary.csv', 'boosting', True)
    # analyze_csv('results-lgbm-boosting-GOOGL-discrete.csv', 'boosting', True)
    # analyze_csv('results-lgbm-walk_forward_test_window_size-GOOGL-binary.csv', 'walk_forward_test_window_size', True)
    # analyze_csv('results-lgbm-walk_forward_test_window_size-GOOGL-discrete.csv', 'walk_forward_test_window_size', True)
    # analyze_final("results-lgbm-final-binary.csv")
    # analyze_final("results-lgbm-final-discrete.csv")
    # analyze_csv('results-rf-pca-GOOGL-binary.csv', 'pca', True)
    # analyze_csv('results-rf-pca-GOOGL-discrete.csv', 'pca', True)
    # analyze_csv('results-rf-n_estimators-GOOGL-binary.csv', 'n_estimators', True)
    # analyze_csv('results-rf-n_estimators-GOOGL-discrete.csv', 'n_estimators', True)
    # analyze_csv('results-rf-walk_forward_test_window_size-GOOGL-binary.csv', 'walk_forward_test_window_size', True)
    # analyze_csv('results-rf-walk_forward_test_window_size-GOOGL-discrete.csv', 'walk_forward_test_window_size', True)
    # analyze_final("results-rf-final-binary.csv")
    # analyze_final("results-rf-final-discrete.csv")
    # analyze_simulation("results-svm-market-simulation-binary.csv", True)
    # analyze_simulation("results-nn-market-simulation-binary.csv")
    # analyze_simulation("results-rf-market-simulation-binary.csv")
    # analyze_simulation("results-lgbm-market-simulation-binary.csv")
    #
    # print('========================================================================================================================================================')
    # analyze_simulation("results-svm-market-simulation-discrete.csv")
    # analyze_simulation("results-nn-market-simulation-discrete.csv")
    # analyze_simulation("results-rf-market-simulation-discrete.csv")
    # analyze_simulation("results-lgbm-market-simulation-discrete.csv")

    # analyze_simulation_details("results-nn-market-simulation-discreteAMGN.csv", 'AMGN', '2019-01-01')
    # analyze_simulation_details("results-svm-market-simulation-discreteGOOGL.csv", 'GOOGL', '2019-01-01')
    # analyze_simulation_details("results-nn-market-simulation-binaryGOOGL.csv", 'GOOGL', '2019-01-01')
    # average_trades(['nn-market-simulation-binary', 'nn-market-simulation-discrete', 'lgbm-market-simulation-binary'
    #                    , 'lgbm-market-simulation-discrete', 'rf-market-simulation-binary', 'rf-market-simulation-discrete'
    #                    , 'svm-market-simulation-binary', 'svm-market-simulation-discrete'])
    print('Result analyzer finished.')
