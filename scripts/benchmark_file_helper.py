import os

from benchmark_params import BenchmarkParams


def initialize_dirs(bench_params: BenchmarkParams):
    if bench_params.save_files and not os.path.exists(bench_params.save_model_path):
        os.makedirs(bench_params.save_model_path)
    if bench_params.save_files and not os.path.exists(bench_params.save_img_path):
        os.makedirs(bench_params.save_img_path)
