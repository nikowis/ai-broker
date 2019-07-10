from benchmark_params import SVMBenchmarkParams, NnBenchmarkParams, LightGBMBenchmarkParams, RandomForestBenchmarkParams
from market_simulation import SVMSimulation, NnMarketSimulation, LightGBMSimulation, RandomForestSimulation
from stock_constants import BASE_COMPANIES

if __name__ == '__main__':
    # bench_params = NnBenchmarkParams(True, benchmark_name='nn-market-simulation-binary')
    # NnMarketSimulation(BASE_COMPANIES, bench_params)
    # bench_params = NnBenchmarkParams(False, benchmark_name='nn-market-simulation-discrete')
    # NnMarketSimulation(BASE_COMPANIES, bench_params)
    # bench_params = LightGBMBenchmarkParams(True, benchmark_name='lgbm-market-simulation-binary')
    # LightGBMSimulation(BASE_COMPANIES, bench_params)
    # bench_params = LightGBMBenchmarkParams(False, benchmark_name='lgbm-market-simulation-discrete')
    # LightGBMSimulation(BASE_COMPANIES, bench_params)
    # bench_params = RandomForestBenchmarkParams(True, benchmark_name='rf-market-simulation-binary')
    # RandomForestSimulation(BASE_COMPANIES, bench_params)
    # bench_params = RandomForestBenchmarkParams(False, benchmark_name='rf-market-simulation-discrete')
    # RandomForestSimulation(BASE_COMPANIES, bench_params)
    # bench_params = SVMBenchmarkParams(True, benchmark_name='svm-market-simulation-binary')
    # SVMSimulation(BASE_COMPANIES, bench_params)
    # bench_params = SVMBenchmarkParams(False, benchmark_name='svm-market-simulation-discrete')
    # SVMSimulation(BASE_COMPANIES, bench_params)

    # bench_params = NnBenchmarkParams(False, benchmark_name='nn-market-simulation-discrete')
    # NnMarketSimulation(BASE_COMPANIES, bench_params, date_simulation_start='2019-03-30',
    #                    date_simulation_end='2019-06-01')

    bench_params = LightGBMBenchmarkParams(False, benchmark_name='lgbm-market-simulation-discrete')
    LightGBMSimulation(BASE_COMPANIES, bench_params, date_simulation_start='2019-03-30',
                       date_simulation_end='2019-06-01')

    # bench_params = SVMBenchmarkParams(False, benchmark_name='svm-market-simulation-discrete')
    # SVMSimulation(BASE_COMPANIES, bench_params, date_simulation_start='2019-03-30', date_simulation_end='2019-06-01')
