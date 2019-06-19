import pandas as pd
import numpy as np

def analyze_csv(filepath, examined_param, print_latex=False):
    df = pd.read_csv(filepath)

    print('Read {0}'.format(filepath))
    df[examined_param] = df[examined_param].astype(str)
    df[examined_param] = df[examined_param].astype(str)
    mean_groupby = df.drop('ID',axis=1).groupby([examined_param]).mean()
    print('Examination of {0}:\n {1}'.format(examined_param, mean_groupby))
    if print_latex:
        print(mean_groupby.to_latex())


if __name__ == '__main__':
    # analyze_csv('nn-pca-GOOGL-binary.csv', 'pca')
    # analyze_csv('results-nn-pca-GOOGL-discrete.csv', 'pca')
    analyze_csv('results-nn-layers-GOOGL-binary.csv', 'layers')
    analyze_csv('results-nn-layers-GOOGL-discrete.csv', 'layers')
