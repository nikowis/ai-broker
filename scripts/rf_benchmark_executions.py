import benchmark_params
from rf_benchmark import RandomForestBenchmark
from stock_constants import BASE_COMPANIES

def rf_examine(binary_classification, examined_params, param_lists, companies=['GOOGL'], walk_forward_testing=False):
    split_params = examined_params.split(',')
    if len(split_params) != len(param_lists):
        print('Examined params length not equal to param lists')
        return

    param_dict = {}
    for i in range(0, len(split_params)):
        examined_param = split_params[i]
        param_list = param_lists[i]
        param_dict.update({examined_param: param_list})

    benchmark_name = 'rf-{0}{1}'.format(''.join(str(p) + "-" for p in split_params),
                                        ''.join(str(c) + "-" for c in companies))
    if binary_classification:
        benchmark_name = benchmark_name + 'binary'
    else:
        benchmark_name = benchmark_name + 'discrete'

    bench_params = benchmark_params.RandomForestBenchmarkParams(binary_classification, examined_param=examined_params,
                                                                benchmark_name=benchmark_name)
    bench_params.plot_partial = True
    bench_params.walk_forward_testing = walk_forward_testing
    RandomForestBenchmark(companies, bench_params, param_dict)


def rf_final(binary):
    benchmark_name = 'rf-final-'
    if binary:
        benchmark_name = benchmark_name + 'binary'
    else:
        benchmark_name = benchmark_name + 'discrete'
    bench_params = benchmark_params.RandomForestBenchmarkParams(binary, benchmark_name=benchmark_name)
    bench_params.plot_partial = False
    benchmark_params.verbose = True
    bench_params.walk_forward_testing = False
    RandomForestBenchmark(BASE_COMPANIES, bench_params)


if __name__ == '__main__':
    #
    # rf_examine(True, 'pca',
    #             [[None, 0.9999, 0.999, 0.99, 0.98, 0.95]])
    # rf_examine(False, 'pca',
    #             [[None, 0.9999, 0.999, 0.99, 0.98, 0.95]])

    # rf_examine(True, 'n_estimators',
    #            [[100, 200, 300, 500, 1000, 1500, 2000, 2500, 3000]])
    # rf_examine(False, 'n_estimators',
    #            [[100, 200, 300, 500, 1000, 1500, 2000]])

    # rf_examine(True, 'max_depth',
    #            [[5, 10, 20, 30, 40, 50, None]])
    # rf_examine(False, 'max_depth',
    #            [[5, 10, 20, 30, 40, 50, None]])

    # rf_examine(True, 'min_samples_leaf',
    #            [[1, 2, 4, 6, 8]])
    # rf_examine(False, 'min_samples_leaf',
    #            [[1, 2, 4, 6, 8]])
    #
    # rf_examine(True, 'min_samples_split',
    #            [[2, 5, 10, 15, 20]])
    # rf_examine(False, 'min_samples_split',
    #            [[2, 5, 10, 15, 20]])

    # rf_examine(True, 'max_features',
    #            [['auto', 'sqrt', 'log2',None]])
    # rf_examine(False, 'max_features',
    #            [['auto', 'sqrt', 'log2',None]])
    # rf_examine(True, 'bootstrap',
    #            [[False, True]])
    # rf_examine(False, 'bootstrap',
    #            [[False, True]])
    # rf_examine(True, 'warm_start',
    #            [[False, True]])
    # rf_examine(False, 'warm_start',
    #            [[False, True]])

    # rf_examine(True, 'walk_forward_test_window_size', [[360, 180, 90, 45, 22]],
    #             walk_forward_testing=True)
    # rf_examine(False, 'walk_forward_test_window_size', [[360, 180, 90, 45, 22]],
    #             walk_forward_testing=True)

    # rf_final(True)
    rf_final(False)

    print('Benchmark executions finished.')
