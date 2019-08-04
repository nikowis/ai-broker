import benchmark_params
from stock_constants import BASE_COMPANIES
from svm_benchmark import SVMBenchmark


def svm_examine(binary_classification, examined_params, param_lists, companies=['GOOGL'], walk_forward_testing=False):
    split_params = examined_params.split(',')
    if len(split_params) != len(param_lists):
        print('Examined params length not equal to param lists')
        return

    param_dict = {}
    for i in range(0, len(split_params)):
        examined_param = split_params[i]
        param_list = param_lists[i]
        param_dict.update({examined_param: param_list})

    benchmark_name = 'svm-{0}{1}'.format(''.join(str(p) + "-" for p in split_params),
                                         ''.join(str(c) + "-" for c in companies))
    if binary_classification:
        benchmark_name = benchmark_name + 'binary'
    else:
        benchmark_name = benchmark_name + 'discrete'

    bench_params = benchmark_params.SVMBenchmarkParams(binary_classification, examined_param=examined_params,
                                                       benchmark_name=benchmark_name)
    bench_params.plot_partial = True
    bench_params.iterations = 1
    bench_params.walk_forward_testing = walk_forward_testing
    SVMBenchmark(companies, bench_params, param_dict)


def svm_final(binary):
    benchmark_name = 'svm-final-'
    if binary:
        benchmark_name = benchmark_name + 'binary'
    else:
        benchmark_name = benchmark_name + 'discrete'
    bench_params = benchmark_params.SVMBenchmarkParams(binary, benchmark_name=benchmark_name)
    bench_params.plot_partial = False
    benchmark_params.verbose = True
    bench_params.walk_forward_testing = False
    SVMBenchmark(BASE_COMPANIES, bench_params)


if __name__ == '__main__':
    # svm_examine(True, 'kernel', [['linear', 'poly', 'rbf', 'sigmoid']])
    # svm_examine(False, 'kernel', [['linear', 'poly', 'rbf', 'sigmoid']])

    # svm_examine(True, 'pca', [[None, 0.9999, 0.999, 0.99, 0.98, 0.97, 0.95, 0.90, 0.85, 0.8]])
    # svm_examine(False, 'pca', [[None, 0.9999, 0.999, 0.99, 0.98, 0.97, 0.95, 0.90, 0.85, 0.8]])
    # svm_examine(True, 'c', [[100, 50, 25, 10, 5, 1, 0.5, 0.1, 0.01, 0.005]])
    # svm_examine(False, 'c', [[100, 50, 25, 10, 5, 1, 0.5, 0.1, 0.01, 0.005]])
    # svm_examine(True, 'gamma', [[0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]])
    # svm_examine(False, 'gamma', [[0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]])
    # svm_examine(True, 'epsilon', [[10, 1, 0.5, 0.1, 0.01, 0.001]])
    # svm_examine(False, 'epsilon', [[10, 1, 0.5, 0.1, 0.01, 0.001]])
    # svm_examine(True, 'walk_forward_test_window_size', [[360, 180, 90, 45, 22]],
    #             walk_forward_testing=True)
    # svm_examine(False, 'walk_forward_test_window_size', [[360, 180, 90, 45, 22]],
    #             walk_forward_testing=True)
    # svm_examine(True, 'walk_forward_test_window_size', [[11, 5, 2, 1]],
    #             walk_forward_testing=True)
    # svm_examine(False, 'walk_forward_test_window_size', [[11, 5, 2, 1]],
    #             walk_forward_testing=True)
    # svm_examine(True, 'max_train_window_size',
    #             [[None, 2000, 1500, 1000, 500, 250, 100]],
    #             walk_forward_testing=True)
    # svm_examine(False, 'max_train_window_size',
    #             [[None, 2000, 1500, 1000, 500, 250, 100]],
    #             walk_forward_testing=True)
    #
    svm_final(True)
    svm_final(False)
    print('Benchmark executions finished.')
