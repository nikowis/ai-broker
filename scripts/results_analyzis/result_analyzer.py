import pandas as pd

ROC_AUC_COL = 'roc_auc'
TRAIN_TIME_COL = 'train_time'
ACCURACY_COL = 'accuracy'


def analyze_csv(filepath, examined_param, print_latex=False):
    df = pd.read_csv(filepath)

    print('Read {0}'.format(filepath))
    df[examined_param] = df[examined_param].astype(str)
    df[examined_param] = df[examined_param].astype(str)
    mean_groupby = df.drop('ID', axis=1).groupby([examined_param]).mean()
    print('Examination of {0}:\n {1}'.format(examined_param, mean_groupby))
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


if __name__ == '__main__':
    # analyze_csv('nn-pca-GOOGL-binary.csv', 'pca')
    # analyze_csv('results-nn-pca-GOOGL-discrete.csv', 'pca')
    # analyze_nn_layers('results-nn-layers-GOOGL-binary.csv')
    # analyze_nn_layers('results-nn-layers-GOOGL-discrete.csv')
    print('Result analyzer finished.')
