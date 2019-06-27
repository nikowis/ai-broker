import pandas as pd

ROC_AUC_COL = 'roc_auc'
TRAIN_TIME_COL = 'train_time'
ACCURACY_COL = 'accuracy'
RESULT_PATH = './../../target/results/'


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
    # analyze_csv('results-svm-kernel-GOOGL-binary.csv', 'kernel', True)
    # analyze_csv('results-svm-kernel-GOOGL-discrete.csv', 'kernel', True)
    # analyze_csv('results-svm-c-kernel-GOOGL-binary.csv', 'c', True)
    # analyze_csv('results-svm-c-kernel-GOOGL-discrete.csv', 'c', True)
    # analyze_csv('results-svm-c-kernel-GOOGL-binary.csv', 'c', True)
    # analyze_csv('results-svm-c-kernel-GOOGL-discrete.csv', 'c', True)
    # analyze_csv('results-svm-walk_forward_test_window_size-GOOGL-binary.csv', 'walk_forward_test_window_size', True)
    # analyze_csv('results-svm-walk_forward_test_window_size-GOOGL-discrete.csv', 'walk_forward_test_window_size', True)
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
    analyze_csv('results-rf-walk_forward_test_window_size-GOOGL-discrete.csv', 'walk_forward_test_window_size', True)
    # analyze_final("results-rf-final-binary.csv")
    # analyze_final("results-rf-final-discrete.csv")
    print('Result analyzer finished.')
