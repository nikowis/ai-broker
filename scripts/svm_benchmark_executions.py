import benchmark_params
from svm_benchmark import SVMBenchmark

SYMBOLS = ['GOOGL', 'MSFT', 'AAPL', 'CSCO', 'INTC', 'FB', 'PEP', 'QCOM', 'AMZN', 'AMGN']


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
    bench_params.walk_forward_testing = walk_forward_testing
    SVMBenchmark(companies, bench_params, param_dict)


def svm_final(binary):
    benchmark_name = 'nn-final-'
    if binary:
        benchmark_name = benchmark_name + 'binary'
    else:
        benchmark_name = benchmark_name + 'discrete'
    bench_params = benchmark_params.SVMBenchmarkParams(binary, benchmark_name=benchmark_name)
    bench_params.plot_partial = False
    benchmark_params.verbose = True
    bench_params.walk_forward_testing = False
    SVMBenchmark(SYMBOLS, bench_params)


if __name__ == '__main__':
    # svm_examine(True, 'kernel', [['linear', 'poly', 'rbf', 'sigmoid']])
    # svm_examine(False, 'kernel', [['linear', 'poly', 'rbf', 'sigmoid']])

    # svm_examine(True, 'c,kernel',
    #             [[25, 10, 5, 1, 0.5, 0.1, 0.01, 0.005], ['linear']])
    # svm_examine(False, 'c,kernel',
    #             [[25, 10, 5, 1, 0.5, 0.1, 0.01, 0.005], ['linear']])
    # svm_examine(True, 'c,gamma,kernel',
    #             [[25, 10, 5, 1, 0.5, 0.1, 0.01, 0.005], [10, 1, 0.5, 0.1, 0.01, 0.001],
    #              ['rbf']])
    # svm_examine(False, 'c,gamma,kernel',
    #             [[25, 10, 5, 1, 0.5, 0.1, 0.01, 0.005], [10, 1, 0.5, 0.1, 0.01, 0.001],
    #              ['rbf']])
    # svm_examine(True, 'pca',
    #             [[None, 0.9999, 0.999, 0.99, 0.98, 0.95]])
    # svm_examine(False, 'pca',
    #             [[None, 0.9999, 0.999, 0.99, 0.98, 0.95]])

    # svm_examine(True, 'walk_forward_test_window_size', [[360, 180, 90, 45, 22]],
    #             walk_forward_testing=True)
    # svm_examine(False, 'walk_forward_test_window_size', [[360, 180, 90, 45, 22]],
    #             walk_forward_testing=True)

    # svm_examine(True, 'walk_forward_test_window_size,c', [[360, 180], [1]],
    #             walk_forward_testing=True)
    # svm_examine(False, 'walk_forward_test_window_size,c', [[360, 180], [1]],
    #             walk_forward_testing=True)

    # svm_examine(True, 'walk_forward_test_window_size,max_train_window_size',
    #             [[90, 45], [2500, 2000, 1500, 1000]],
    #             walk_forward_testing=True)
    # svm_examine(False, 'walk_forward_test_window_size,max_train_window_size',
    #             [[90, 45], [2500, 2000, 1500, 1000]],
    #             walk_forward_testing=True)
    #
    svm_final(True)
    svm_final(False)
    print('Benchmark executions finished.')
