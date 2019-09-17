from benchmark_params import NnBenchmarkParams, LightGBMBenchmarkParams
from market_simulation import NnMarketSimulation, LightGBMSimulation
from stock_constants import BASE_COMPANIES

if __name__ == '__main__':
    # bench_params = NnBenchmarkParams(True, benchmark_name='nn-market-simulation-binary-walk-forward-5')
    # NnMarketSimulation(BASE_COMPANIES, bench_params)

    # for i in range(0, len(BASE_COMPANIES)):
    #     bench_params = NnBenchmarkParams(True, benchmark_name='nn-market-simulation-binary-walk-forward-2-it-' + str(i))
    #     bench_params.walk_forward_test_window_size=2
    #     NnMarketSimulation([BASE_COMPANIES[i]], bench_params)
    #     print(datetime.datetime.now())
    # for i in range(3, 4):
    #     bench_params = NnBenchmarkParams(False, benchmark_name='nn-market-simulation-discrete-walk-forward-2-it-' + str(i))
    #     bench_params.walk_forward_test_window_size=2
    #     NnMarketSimulation([BASE_COMPANIES[i]], bench_params)
    #     print(datetime.datetime.now())

    bench_params = LightGBMBenchmarkParams(
        binary_classification=True
        , benchmark_name='lgbm-market-simulation-binary'
    )
    LightGBMSimulation(
        ['GOOGL', 'MSFT', 'AAPL', 'CSCO', 'INTC', 'FB', 'PEP', 'QCOM', 'AMZN', 'AMGN']
        , bench_params
        , date_simulation_start='2019-08-01'
        , date_simulation_end='2019-09-01'
        , budget=1000
    )



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

    # bench_params = SVMBenchmarkParams(True, benchmark_name='svm-market-simulation-binary-tough-times')
    # SVMSimulation(BASE_COMPANIES, bench_params, date_simulation_start='2019-03-30', date_simulation_end='2019-06-01')

    # bench_params = LightGBMBenchmarkParams(True, benchmark_name='lgbm-market-simulation-binary-crisis')
    # LightGBMSimulation(['INTC'], bench_params, date_simulation_start='2008-01-01', date_simulation_end='2009-01-01')
    # bench_params = LightGBMBenchmarkParams(False, benchmark_name='lgbm-market-simulation-discrete-crisis')
    # LightGBMSimulation(['INTC'], bench_params, date_simulation_start='2008-01-01', date_simulation_end='2009-01-01')
    bench_params = LightGBMBenchmarkParams(True, benchmark_name='lgbm-market-simulation-binary-2k19')
    LightGBMSimulation(['GOOGL'], bench_params, date_simulation_start='2019-01-01', date_simulation_end='2019-07-1')
    bench_params = LightGBMBenchmarkParams(False, benchmark_name='lgbm-market-simulation-discrete-2k19')
    LightGBMSimulation(['GOOGL'], bench_params, date_simulation_start='2019-01-01', date_simulation_end='2019-07-1')

    # bench_params = NnBenchmarkParams(True, benchmark_name='nn-market-simulation-binary-crisis')
    # bench_params.walk_forward_test_window_size = 2
    # NnMarketSimulation(['INTC'], bench_params, date_simulation_start='2008-01-01', date_simulation_end='2009-01-01')
    # bench_params = NnBenchmarkParams(False, benchmark_name='nn-market-simulation-discrete-crisis')
    # bench_params.walk_forward_test_window_size = 2
    # NnMarketSimulation(['INTC'], bench_params, date_simulation_start='2008-01-01', date_simulation_end='2009-01-01')

    # bench_params = NnBenchmarkParams(True, benchmark_name='nn-market-simulation-binary-summer')
    # bench_params.walk_forward_test_window_size = 2
    # NnMarketSimulation(['INTC'], bench_params, date_simulation_start='2019-06-01', date_simulation_end='2019-08-14')
    # bench_params = NnBenchmarkParams(False, benchmark_name='nn-market-simulation-discrete-summer')
    # bench_params.walk_forward_test_window_size = 2
    # NnMarketSimulation(['INTC'], bench_params, date_simulation_start='2019-06-01', date_simulation_end='2019-08-14')
