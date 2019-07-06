from benchmark_params import LightGBMBenchmarkParams, RandomForestBenchmarkParams, \
    SVMBenchmarkParams

SYMBOLS = ['GOOGL', 'MSFT', 'AAPL', 'CSCO', 'INTC', 'FB', 'PEP', 'QCOM', 'AMZN', 'AMGN']

from market_simulation import LightGBMSimulation, RandomForestSimulation, SVMSimulation

if __name__ == '__main__':
    # bench_params = NnBenchmarkParams(True, benchmark_name='nn-market-simulation-binary')
    # NnMarketSimulation(SYMBOLS, bench_params)
    # bench_params = NnBenchmarkParams(False, benchmark_name='nn-market-simulation-discrete')
    # NnMarketSimulation(SYMBOLS, bench_params)
    bench_params = LightGBMBenchmarkParams(True, benchmark_name='lgbm-market-simulation-binary')
    LightGBMSimulation(SYMBOLS, bench_params)
    bench_params = LightGBMBenchmarkParams(False, benchmark_name='lgbm-market-simulation-discrete')
    LightGBMSimulation(SYMBOLS, bench_params)
    bench_params = RandomForestBenchmarkParams(True, benchmark_name='rf-market-simulation-binary')
    RandomForestSimulation(SYMBOLS, bench_params)
    bench_params = RandomForestBenchmarkParams(False, benchmark_name='rf-market-simulation-discrete')
    RandomForestSimulation(SYMBOLS, bench_params)
    bench_params = SVMBenchmarkParams(True, benchmark_name='svm-market-simulation-binary')
    SVMSimulation(SYMBOLS, bench_params)
    bench_params = SVMBenchmarkParams(False, benchmark_name='svm-market-simulation-discrete')
    SVMSimulation(SYMBOLS, bench_params)
