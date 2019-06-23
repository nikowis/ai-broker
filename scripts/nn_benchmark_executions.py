import benchmark_params
from nn_benchmark import NnBenchmark

SYMBOLS = ['GOOGL', 'MSFT', 'AAPL', 'CSCO', 'INTC', 'FB', 'PEP', 'QCOM', 'AMZN', 'AMGN']


def nn_examine(binary_classification, examined_params, param_lists, companies=['GOOGL'], walk_forward_testing=False):
    split_params = examined_params.split(',')
    if len(split_params) != len(param_lists):
        print('Examined params length not equal to param lists')
        return

    param_dict = {}
    for i in range(0, len(split_params)):
        examined_param = split_params[i]
        param_list = param_lists[i]
        param_dict.update({examined_param: param_list})

    benchmark_name = 'nn-{0}{1}'.format(''.join(str(p) + "-" for p in split_params),
                                        ''.join(str(c) + "-" for c in companies))
    if binary_classification:
        benchmark_name = benchmark_name + 'binary'
    else:
        benchmark_name = benchmark_name + 'discrete'

    bench_params = benchmark_params.NnBenchmarkParams(binary_classification, examined_param=examined_params,
                                                      benchmark_name=benchmark_name)
    bench_params.plot_partial = True
    bench_params.walk_forward_testing = walk_forward_testing
    NnBenchmark(companies, bench_params, param_dict)


def nn_final_binary():
    benchmark_name = 'nn-final-binary'
    bench_params = benchmark_params.NnBenchmarkParams(True, benchmark_name=benchmark_name)
    bench_params.plot_partial = False
    benchmark_params.verbose = True
    bench_params.walk_forward_testing = True
    bench_params.iterations = 3
    NnBenchmark(SYMBOLS, bench_params)


def nn_final_discrete():
    for i in range(2, len(SYMBOLS)):
        sym = SYMBOLS[i]
        benchmark_name = 'nn-final-discrete-' + str(i)
        bench_params = benchmark_params.NnBenchmarkParams(False, benchmark_name=benchmark_name)
        bench_params.plot_partial = False
        benchmark_params.verbose = True
        bench_params.walk_forward_testing = True
        bench_params.iterations = 3
        NnBenchmark([sym], bench_params)


if __name__ == '__main__':
    # nn_examine(True, 'pca', [[None, 0.9999, 0.999, 0.99, 0.98, 0.97]])
    # nn_examine(False, 'pca', [[None, 0.9999, 0.999, 0.99, 0.98, 0.97]])
    # nn_examine(True, 'regularizer', [[None, 0.005, 0.01, 0.02]])
    # nn_examine(False, 'regularizer', [[None, 0.005, 0.01, 0.02]])
    # nn_examine(True, 'layers', [[[], [2], [3], [4], [5], [6], [7], [8], [9], [10], [2, 2], [3, 3], [4, 4], [5, 5],
    #                              [6, 6], [7, 7], [8, 8], [9, 9], [10, 10]]])

    # nn_examine(True, 'batch_size', [[1, 5, 10, 20, 40]])
    # nn_examine(False, 'batch_size', [[1, 5, 10, 20, 40]])
    # nn_examine(False, 'max_train_window_size', [[None, 2400, 2000, 1500, 1000]])
    # nn_examine(True, 'max_train_window_size', [[None, 2400, 2000, 1500, 1000]])
    # nn_examine(True, 'walk_forward_test_window_size', [[360, 180, 90, 45, 22]],
    #            walk_forward_testing=True)
    # nn_examine(False, 'walk_forward_test_window_size', [[360, 180, 90, 45, 22]],
    #            walk_forward_testing=True)
    # nn_final_binary()
    nn_final_discrete()
    print('Benchmark executions finished.')
