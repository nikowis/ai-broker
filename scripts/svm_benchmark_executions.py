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


if __name__ == '__main__':
    # svm_examine(True, 'kernel', [['linear', 'poly', 'rbf', 'sigmoid']])
    # svm_examine(False, 'kernel', [['linear', 'poly', 'rbf', 'sigmoid']])
    svm_examine(False, 'c,epsilon,kernel',
                [[25, 10, 5, 1, 0.5, 0.1, 0.01, 0.005], [0.2, 0.15, 0.1, 0.05, 0.025], ['linear']])
    svm_examine(True, 'c,epsilon,kernel',
                [[25, 10, 5, 1, 0.5, 0.1, 0.01, 0.005], [0.2, 0.15, 0.1, 0.05, 0.025], ['linear']])
    print('Benchmark executions finished.')
