import benchmark_params
from benchmark import NnBenchmark

SYMBOLS = ['GOOGL', 'MSFT', 'AAPL', 'CSCO', 'INTC', 'FB', 'PEP', 'QCOM', 'AMZN', 'AMGN']


def nn_pca_GOOGL():
    bench_params = benchmark_params.NnBenchmarkParams(True, examined_param='pca', benchmark_name='nn-pca-GOOGL-binary')
    bench_params.plot_partial = True
    NnBenchmark(['GOOGL'], bench_params, {'pca': [None, 0.999, 0.99, 0.97, 0.95, 0.90, 0.80]})


def nn_pca_GOOGL_discrete():
    bench_params = benchmark_params.NnBenchmarkParams(False, examined_param='pca',
                                                      benchmark_name='nn-pca-GOOGL-discrete')
    bench_params.plot_partial = True
    NnBenchmark(['GOOGL'], bench_params, {'pca': [None, 0.999, 0.99, 0.97, 0.95, 0.90, 0.80]})


def nn_layers_GOOGL():
    bench_params = benchmark_params.NnBenchmarkParams(True, examined_param='layers',
                                                      benchmark_name='nn-layers-GOOGL-binary')
    bench_params.plot_partial = True
    NnBenchmark(['GOOGL'], bench_params, {
        'layers': [[], [2], [3], [4], [5], [6], [7], [8], [9], [10], [2, 2], [3, 3], [4, 4], [5, 5],
                   [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5],
                   [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10]]})


def nn_layers_GOOGL_discrete():
    bench_params = benchmark_params.NnBenchmarkParams(False, examined_param='layers',
                                                      benchmark_name='nn-layers-GOOGL-discrete')
    bench_params.plot_partial = True
    NnBenchmark(['GOOGL'], bench_params, {
        'layers': [[], [2], [3], [4], [5], [6], [7], [8], [9], [10], [2, 2], [3, 3], [4, 4], [5, 5],
                   [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5],
                   [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10]]})


if __name__ == '__main__':
    # nn_pca_GOOGL()
    # nn_pca_GOOGL_discrete()
    # nn_layers_GOOGL()
    # nn_layers_GOOGL_discrete()
    print('Benchmark executions finished.')
