import benchmark_params
from benchmark import NnBenchmark

SYMBOLS = ['GOOGL', 'MSFT', 'AAPL', 'CSCO', 'ORCL', 'INTC', 'VOD', 'QCOM', 'AMZN', 'AMGN']


def nn_pca_10_companies():
    bench_params = benchmark_params.NnBenchmarkParams(True, examined_param='pca', benchmark_name='nn-pca-GOOGL-second')
    bench_params.plot_partial = True
    NnBenchmark(['GOOGL'], bench_params, {'pca': [None, 0.999, 0.99, 0.97, 0.95, 0.90, 0.80]})


if __name__ == '__main__':
    nn_pca_10_companies()