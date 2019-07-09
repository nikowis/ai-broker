import benchmark_params
from lgbm_benchmark import LightGBMBenchmark

from stock_constants import BASE_COMPANIES

def lgbm_examine(binary_classification, examined_params, param_lists, companies=['GOOGL'], walk_forward_testing=False):
    split_params = examined_params.split(',')
    if len(split_params) != len(param_lists):
        print('Examined params length not equal to param lists')
        return

    param_dict = {}
    for i in range(0, len(split_params)):
        examined_param = split_params[i]
        param_list = param_lists[i]
        param_dict.update({examined_param: param_list})

    benchmark_name = 'lgbm-{0}{1}'.format(''.join(str(p) + "-" for p in split_params),
                                          ''.join(str(c) + "-" for c in companies))
    if binary_classification:
        benchmark_name = benchmark_name + 'binary'
    else:
        benchmark_name = benchmark_name + 'discrete'

    bench_params = benchmark_params.LightGBMBenchmarkParams(binary_classification, examined_param=examined_params,
                                                            benchmark_name=benchmark_name)
    bench_params.plot_partial = True
    bench_params.walk_forward_testing = walk_forward_testing
    LightGBMBenchmark(companies, bench_params, param_dict)


def lgbm_final(binary):
    benchmark_name = 'lgbm-final-'
    if binary:
        benchmark_name = benchmark_name + 'binary'
    else:
        benchmark_name = benchmark_name + 'discrete'
    bench_params = benchmark_params.LightGBMBenchmarkParams(binary, benchmark_name=benchmark_name)
    bench_params.plot_partial = False
    benchmark_params.verbose = True
    bench_params.walk_forward_testing = False
    LightGBMBenchmark(BASE_COMPANIES, bench_params)


if __name__ == '__main__':
    # lgbm_examine(True, 'pca', [[None, 0.9999, 0.999, 0.99, 0.9]])
    # lgbm_examine(False, 'pca', [[None, 0.9999, 0.999, 0.99, 0.9]])
    # lgbm_examine(True, 'feature_fraction', [[1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]])
    # lgbm_examine(False, 'feature_fraction',[[1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]])
    # lgbm_examine(True, 'boosting', [['dart', 'gbdt', 'gbrt','goss']])
    # lgbm_examine(False, 'boosting', [['dart', 'gbdt', 'gbrt','goss']])
    # lgbm_examine(True, 'max_depth', [[-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ,15]])
    # lgbm_examine(False, 'max_depth', [[-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ,15]])
    # lgbm_examine(True, 'num_leaves', [[5, 10, 20, 30, 50, 60, 70, 80, 90, 100, 200]])
    # lgbm_examine(False, 'num_leaves', [[5, 10, 20, 30, 50, 60, 70, 80, 90, 100, 20]])
    # lgbm_examine(True, 'max_bin', [[10, 50, 100, 200, 300, 500, 700, 1000, 1200, 1400, 1600]])
    # lgbm_examine(False, 'max_bin', [[10, 50, 100, 200, 300, 500, 700, 1000, 1200, 1400, 1600]])
    # lgbm_examine(True, 'min_sum_hessian_in_leaf', [[0.001, 0.01, 0.1, 1, 2, 5 ,8, 10, 15, 25, 50, 100]])
    # lgbm_examine(False, 'min_sum_hessian_in_leaf', [[0.001, 0.01, 0.1, 1, 2, 5 ,8, 10, 15, 25, 50, 100]])
    # lgbm_examine(True, 'min_data_in_leaf', [[1, 3, 5, 10, 15, 20, 25, 50, 100]])
    # lgbm_examine(False, 'min_data_in_leaf', [[1, 3, 5, 10, 15, 20, 25, 50, 100]])

    # lgbm_examine(True, 'walk_forward_test_window_size', [[360, 180, 90, 45, 22]],
    #             walk_forward_testing=True)
    # lgbm_examine(False, 'walk_forward_test_window_size', [[360, 180, 90, 45, 22]],
    #             walk_forward_testing=True)
    # lgbm_final(True)
    lgbm_final(False)
    print('Benchmark executions finished.')
