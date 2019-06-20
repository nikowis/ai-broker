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


def nn_test_windows_size_GOOGL():
    bench_params = benchmark_params.NnBenchmarkParams(True, examined_param='walk_forward_test_window_size',
                                                      benchmark_name='nn-test-windows-size-GOOGL-binary-2')
    bench_params.walk_forward_testing = True
    bench_params.epochs = 50
    bench_params.walk_forward_retrain_epochs = 1
    bench_params.plot_partial = True
    NnBenchmark(['GOOGL'], bench_params, {'walk_forward_learn_from_scratch': [False, True],
                                          'walk_forward_test_window_size': [720, 600, 500, 400, 360, 300, 200, 150, 100,
                                                                            90]})


def nn_test_windows_size_GOOGL_discrete():
    bench_params = benchmark_params.NnBenchmarkParams(False, examined_param='walk_forward_test_window_size',
                                                      benchmark_name='nn-test-windows-size-GOOGL-discrete')
    bench_params.walk_forward_testing = True
    bench_params.epochs = 50
    bench_params.walk_forward_retrain_epochs = 3
    bench_params.plot_partial = True
    NnBenchmark(['GOOGL'], bench_params, {'walk_forward_learn_from_scratch': [False, True],
                                          'walk_forward_test_window_size': [720, 600, 500, 400, 360, 300, 200, 150, 100,
                                                                            90]})


def nn_walk_forward_retrain_epochs_GOOGL():
    bench_params = benchmark_params.NnBenchmarkParams(True, examined_param='walk_forward_retrain_epochs',
                                                      benchmark_name='nn-walk_forward_retrain_epochs-GOOGL-binary')
    bench_params.walk_forward_testing = True
    bench_params.epochs = 50
    bench_params.walk_forward_test_window_size = 180
    bench_params.plot_partial = True
    NnBenchmark(['GOOGL'], bench_params, {
        'walk_forward_retrain_epochs': [1, 2, 3, 4, 5, 7, 9, 11]})


def nn_walk_forward_retrain_epochs_GOOGL_discrete():
    bench_params = benchmark_params.NnBenchmarkParams(False, examined_param='walk_forward_retrain_epochs',
                                                      benchmark_name='nn-walk_forward_retrain_epochs-GOOGL-discrete')
    bench_params.walk_forward_testing = True
    bench_params.epochs = 50
    bench_params.walk_forward_test_window_size = 180
    bench_params.plot_partial = True
    NnBenchmark(['GOOGL'], bench_params, {
        'walk_forward_retrain_epochs': [1, 2, 3, 4, 5, 7, 9, 11]})


if __name__ == '__main__':
    # nn_pca_GOOGL()
    # nn_pca_GOOGL_discrete()
    # nn_layers_GOOGL()
    # nn_layers_GOOGL_discrete()
    nn_test_windows_size_GOOGL()
    nn_test_windows_size_GOOGL_discrete()
    # nn_walk_forward_retrain_epochs_GOOGL()
    # nn_walk_forward_retrain_epochs_GOOGL_discrete()
    print('Benchmark executions finished.')
