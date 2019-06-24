import benchmark_params
from lgbm_benchmark import LightGBMBenchmark

SYMBOLS = ['GOOGL', 'MSFT', 'AAPL', 'CSCO', 'INTC', 'FB', 'PEP', 'QCOM', 'AMZN', 'AMGN']


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
    LightGBMBenchmark(SYMBOLS, bench_params)


if __name__ == '__main__':
    print('Benchmark executions finished.')
