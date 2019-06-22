import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import style
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import csv_importer
import stock_constants as const

MIN_DATE = '1900-01-01'
MAX_DATE = '2020-10-29'
SELECTED_SYM = 'GOOGL'
IMG_PATH = './../../target/data_analyzis/'
if not os.path.exists(IMG_PATH):
    os.makedirs(IMG_PATH)

CORRELATED_COLS = [const.APO_10_COL, const.APO_DIFF_COL, const.MOM_5_COL, const.MOM_10_COL, const.MOM_DIFF_COL,
                   const.ROC_5_COL,
                   const.ROC_10_COL, const.RSI_10_COL]

HELPER_COLS = [const.LABEL_COL, const.LABEL_BINARY_COL, const.LABEL_DISCRETE_COL, const.DAILY_PCT_CHANGE_COL,
               const.DIVIDENT_AMOUNT_COL, const.SPLIT_COEFFICIENT_COL, const.CLOSE_COL, const.BBANDS_10_RLB_COL,
               const.BBANDS_10_RMB_COL, const.BBANDS_10_RUB_COL, const.BBANDS_20_RLB_COL, const.BBANDS_20_RMB_COL,
               const.BBANDS_20_RUB_COL, const.MACD_HIST_COL]


def plot_columns(df):
    style.use('ggplot')

    line_plot_column(df, const.ADJUSTED_CLOSE_COL, 'GOOGL', 'Cena zamknięcia (USD)', 'Data')
    df[const.ADJUSTED_CLOSE_COL + ' stationary'] = df[const.ADJUSTED_CLOSE_COL].diff().fillna(0)
    line_plot_column(df, const.ADJUSTED_CLOSE_COL + ' stationary', SELECTED_SYM, 'Cena zamknięcia (USD)', 'Data')
    line_plot_column(df, const.VOLUME_COL, SELECTED_SYM, 'Liczba akcji w obrocie', 'Data')
    line_plot_column(df, const.HL_PCT_CHANGE_COL, SELECTED_SYM, 'Stosunek high/low (%)', 'Data')
    line_plot_column(df, const.SMA_5_COL, SELECTED_SYM, 'SMA-5', 'Data')
    line_plot_column(df, const.SMA_DIFF_COL, SELECTED_SYM, 'SMA-DIFF', 'Data')
    line_plot_column(df, const.TR_COL, SELECTED_SYM, 'TR', 'Data')
    line_plot_column(df, const.MACD_COL, SELECTED_SYM, 'MACD', 'Data')
    line_plot_column(df, const.MACD_SIGNAL_COL, SELECTED_SYM, 'MACD Signal', 'Data')

    line_plot_column(df, const.ROC_5_COL, SELECTED_SYM, 'ROC-5', 'Data')
    line_plot_column(df, const.ROC_DIFF_COL, SELECTED_SYM, 'ROC-DIFF', 'Data')

    line_plot_column(df, const.MOM_5_COL, SELECTED_SYM, 'MOM-5', 'Data')
    line_plot_column(df, const.MOM_DIFF_COL, SELECTED_SYM, 'MOM-DIFF', 'Data')

    line_plot_column(df, const.WILLR_5_COL, SELECTED_SYM, 'WILLR-5', 'Data')
    line_plot_column(df, const.WILLR_DIFF_COL, SELECTED_SYM, 'WILLR-DIFF', 'Data')

    line_plot_column(df, const.RSI_5_COL, SELECTED_SYM, 'RSI-5', 'Data')
    line_plot_column(df, const.RSI_DIFF_COL, SELECTED_SYM, 'RSI-DIFF', 'Data')

    line_plot_column(df, const.ADX_5_COL, SELECTED_SYM, 'ADX-5', 'Data')
    line_plot_column(df, const.ADX_DIFF_COL, SELECTED_SYM, 'ADX-DIFF', 'Data')

    line_plot_column(df, const.CCI_5_COL, SELECTED_SYM, 'CCI-5', 'Data')
    line_plot_column(df, const.CCI_DIFF_COL, SELECTED_SYM, 'CCI-DIFF', 'Data')

    line_plot_column(df, const.AD_COL, SELECTED_SYM, 'AD', 'Data')
    line_plot_column(df, const.STOCH_K_COL, SELECTED_SYM, 'STOCH %K', 'Data')
    line_plot_column(df, const.STOCH_D_COL, SELECTED_SYM, 'STOCH %D', 'Data')
    line_plot_column(df, const.STOCH_K_DIFF_COL, SELECTED_SYM, 'STOCH %K DIFF', 'Data')
    line_plot_column(df, const.STOCH_D_DIFF_COL, SELECTED_SYM, 'STOCH %D DIFF', 'Data')
    line_plot_column(df, const.DISPARITY_5_COL, SELECTED_SYM, 'DISPARITY 5', 'Data')
    line_plot_column(df, const.BBANDS_10_DIFF_COL, SELECTED_SYM, 'BBANDS 10 DIFF', 'Data')
    line_plot_column(df, const.PRICE_BBANDS_UP_10_COL, SELECTED_SYM, 'PRICE TO BBADS UP 10', 'Data')
    line_plot_column(df, const.PRICE_BBANDS_LOW_10_COL, SELECTED_SYM, 'PRICE TO BBADS LOW 10', 'Data')

    hist_plot_column(df, const.LABEL_BINARY_COL, SELECTED_SYM, 'Liczba przypadków', 'Trend', [-1, 1],
                     ['Maleje', 'Rośnie'])
    hist_plot_column(df, const.LABEL_DISCRETE_COL, SELECTED_SYM, 'Liczba przypadków', 'Trend', [-1, 0, 1],
                     ['Maleje', 'Utrzymuje się', 'Rośnie'])


def line_plot_column(df, colname, title, ylabel, xlabel):
    df[colname].plot(kind='line')
    describe_plot_and_save(colname, title, ylabel, xlabel)


def hist_plot_column(df, colname, title, ylabel, xlabel, xticks, tick_labels):
    df[colname].plot(kind='hist', xticks=xticks)
    plt.xticks(xticks, tick_labels)
    describe_plot_and_save(colname, title, ylabel, xlabel)


def describe_plot_and_save(colname, title, ylabel, xlabel):
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig('{}/{}.png'.format(IMG_PATH, colname))
    plt.savefig('{}/{}.pdf'.format(IMG_PATH, colname), format='pdf', dpi=1000)
    plt.close()


def describe_df():
    feature_names = list(df.columns.values)
    describe = df.describe(percentiles=[.01, 0.05, .5, .95, .99]).T
    describe = describe.drop(columns=['count'])
    describe.insert(loc=0, column='feature', value=feature_names)
    pd.options.display.float_format = '{:5,.2f}'.format
    print(describe.to_latex(index=False, longtable=True, ))


def count_outliers(df):
    df[const.ADJUSTED_CLOSE_COL] = df[const.ADJUSTED_CLOSE_COL].diff()
    df[const.OPEN_COL] = df[const.OPEN_COL].diff()
    df[const.CLOSE_COL] = df[const.CLOSE_COL].diff()
    df[const.HIGH_COL] = df[const.HIGH_COL].diff()
    df[const.LOW_COL] = df[const.LOW_COL].diff()
    df[const.SMA_5_COL] = df[const.SMA_5_COL].diff()
    df[const.SMA_10_COL] = df[const.SMA_10_COL].diff()
    df[const.SMA_20_COL] = df[const.SMA_20_COL].diff()

    df.dropna(inplace=True)
    df_without_helper_cols = df.drop(
        HELPER_COLS, axis=1)

    df_without_corelated_features = df_without_helper_cols.drop(CORRELATED_COLS, axis=1)

    Q1 = df_without_corelated_features.quantile(0.1)
    Q3 = df_without_corelated_features.quantile(0.9)
    IQR = Q3 - Q1
    result = ((df_without_corelated_features < (Q1 - 1.5 * IQR)) | (
            df_without_corelated_features > (Q3 + 1.5 * IQR))).sum()

    row_count = len(df.index)

    # result = result.add(pd.Series(row_count))
    print(result)
    print('Total rows ', row_count)


def plot_correlations(df_without_corelated_features, img_path='./../target/documentation_plots_and_images'):
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    res = get_top_abs_correlations(df_without_corelated_features, 30)
    print('Most correalated features:')
    print(pd.DataFrame(res).to_latex())
    corr = df_without_corelated_features.corr()
    # sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
    #            square=True, ax=ax)
    fig, ax = plt.subplots(figsize=(11, 10))
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                ax=ax)
    fig.tight_layout()
    plt.savefig('{}/{}.pdf'.format(img_path, 'corr_matrix'), format='pdf', dpi=1000)
    plt.savefig('{}/{}.png'.format(img_path, 'corr_matrix'))
    plt.show()
    plt.close()


def principal_component_analysis(x, img_path='./../target/documentation_plots_and_images'):
    pca = PCA().fit(x)
    pca_95 = PCA(.95).fit(x)

    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.title('PCA')
    plt.xlabel('Liczba komponentów')
    plt.ylabel('Suma wyjaśnionej wariancji')
    plt.savefig('{}/{}.pdf'.format(img_path, 'pca_variance'), format='pdf', dpi=1000)
    plt.savefig('{}/{}.png'.format(img_path, 'pca_variance'))
    plt.show()
    plt.close()
    explained = np.cumsum(pca.explained_variance_ratio_)
    print(explained)
    print(pca.n_components_, ' components')
    x = pca_95.transform(x)
    return x


def pca_vs_columns_len(df, pcas):
    results_dict = {'pca': [], 'components': []}
    results_df = pd.DataFrame(
        data=results_dict)

    df[const.ADJUSTED_CLOSE_COL] = df[const.ADJUSTED_CLOSE_COL].diff()
    df[const.OPEN_COL] = df[const.OPEN_COL].diff()
    df[const.CLOSE_COL] = df[const.CLOSE_COL].diff()
    df[const.HIGH_COL] = df[const.HIGH_COL].diff()
    df[const.LOW_COL] = df[const.LOW_COL].diff()
    df[const.SMA_5_COL] = df[const.SMA_5_COL].diff()
    df[const.SMA_10_COL] = df[const.SMA_10_COL].diff()
    df[const.SMA_20_COL] = df[const.SMA_20_COL].diff()
    df.dropna(inplace=True)
    df_without_helper_cols = df.drop(HELPER_COLS, axis=1)
    df_without_corelated_features = df_without_helper_cols.drop(CORRELATED_COLS, axis=1)
    x = np.array(df_without_corelated_features)
    y = np.array(df[const.LABEL_BINARY_COL])
    x_train, _, _, _ = model_selection.train_test_split(x, y, test_size=0.8, shuffle=False)
    scale = StandardScaler().fit(x_train)
    x_train = scale.transform(x_train)
    results_dict = {'pca': None, 'components': x_train.shape[1]}
    results_df = results_df.append(results_dict, ignore_index=True)
    for p in pcas:
        pca = PCA(p).fit(x_train)
        x_train = pca.transform(x_train)
        results_dict = {'pca': p, 'components': x_train.shape[1]}
        results_df = results_df.append(results_dict, ignore_index=True)

    print(results_df.to_latex(index=False))


def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i + 1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]


if __name__ == '__main__':
    df_list, sym_list = csv_importer.import_data_from_files([SELECTED_SYM], './../../target/data/')

    df = df_list[0]
    # describe_df()
    # plot_columns(df)
    pca_vs_columns_len(df, [0.9999, 0.999, 0.99, 0.98, 0.97])
