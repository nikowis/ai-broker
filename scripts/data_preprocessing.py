import matplotlib.pyplot as plt
from matplotlib import style
import os
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import db_access
import stock_constants as const
import pandas as pd
import numpy as np

MIN_DATE = '2019-01-01'
MAX_DATE = '2020-10-29'
SELECTED_SYM = 'GOOGL'
IMG_PATH = './../target/documentation_plots_and_images/'
if not os.path.exists(IMG_PATH):
    os.makedirs(IMG_PATH)


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


def preprocess(df, standarize=True, difference_non_stationary=True, binary_classification=True):
    df[const.ADJUSTED_CLOSE_COL] = df[const.ADJUSTED_CLOSE_COL].diff()
    df[const.OPEN_COL] = df[const.OPEN_COL].diff()
    df[const.CLOSE_COL] = df[const.CLOSE_COL].diff()
    df[const.HIGH_COL] = df[const.HIGH_COL].diff()
    df[const.LOW_COL] = df[const.LOW_COL].diff()
    df[const.SMA_5_COL] = df[const.SMA_5_COL].diff()
    df[const.SMA_10_COL] = df[const.SMA_10_COL].diff()
    df[const.SMA_20_COL] = df[const.SMA_20_COL].diff()
    df.dropna(inplace=True)
    df_without_helper_cols = df.drop([const.LABEL_COL, const.LABEL_BINARY_COL, const.LABEL_DISCRETE_COL, const.DAILY_PCT_CHANGE_COL,
                                      const.DIVIDENT_AMOUNT_COL, const.SPLIT_COEFFICIENT_COL, const.CLOSE_COL,
                                      const.BBANDS_10_RLB_COL, const.BBANDS_10_RMB_COL, const.BBANDS_10_RUB_COL,
                                      const.BBANDS_20_RLB_COL, const.BBANDS_20_RMB_COL, const.BBANDS_20_RUB_COL,
                                      const.MACD_HIST_COL], axis=1)

    df_without_corelated_features = df_without_helper_cols.drop([const.APO_10_COL, const.APO_DIFF_COL,
                                                                 const.MOM_5_COL, const.MOM_10_COL, const.MOM_DIFF_COL,
                                                                 const.ROC_5_COL, const.ROC_10_COL, const.RSI_10_COL], axis=1)

    if binary_classification:
        y = np.array(df[const.LABEL_BINARY_COL])
    else:
        y = np.array(df[const.LABEL_DISCRETE_COL])

    x = np.array(df_without_corelated_features)
    if standarize:
        std_scale = StandardScaler().fit(x)
        x = std_scale.transform(x)

    res = get_top_abs_correlations(df_without_corelated_features, 30)

    print('Most correalated features:')
    print(pd.DataFrame(res).to_latex())

    ##Corr Matrix
    corr = df_without_corelated_features.corr()

    # sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
    #            square=True, ax=ax)

    fig, ax = plt.subplots(figsize=(11, 10))

    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                cmap = sns.diverging_palette(220, 10, as_cmap=True),
                ax=ax)
    fig.tight_layout()

    plt.savefig('{}/{}.pdf'.format(IMG_PATH, 'corr_matrix'), format='pdf', dpi=1000)
    plt.savefig('{}/{}.png'.format(IMG_PATH, 'corr_matrix'))
    # plt.show()
    plt.close()
    #End Corr Matrix

    pca = PCA().fit(x)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.title('PCA')
    plt.xlabel('Liczba komponentów')
    plt.ylabel('Suma wyjaśnionej wariancji')
    plt.savefig('{}/{}.pdf'.format(IMG_PATH, 'pca_variance'), format='pdf', dpi=1000)
    plt.savefig('{}/{}.png'.format(IMG_PATH, 'pca_variance'))
    # plt.show()
    plt.close()
    return df, x, y


if __name__ == '__main__':
    db_conn = db_access.create_db_connection(remote=False)
    df_list, sym_list = db_access.find_by_tickers_to_dateframe_parse_to_df_list(db_conn, [SELECTED_SYM],
                                                                                min_date=MIN_DATE, max_date=MAX_DATE)
    df = df_list[0]
    preprocess(df)
