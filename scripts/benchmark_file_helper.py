import os
import re
from shutil import copyfile

from benchmark_params import BenchmarkParams


def initialize_dirs(bench_params: BenchmarkParams):
    if bench_params.save_files and not os.path.exists(bench_params.save_model_path):
        os.makedirs(bench_params.save_model_path)
    if bench_params.save_files and not os.path.exists(bench_params.save_img_path):
        os.makedirs(bench_params.save_img_path)
    if bench_params.save_files and bench_params.plot_partial and not os.path.exists(bench_params.save_partial_img_path):
        os.makedirs(bench_params.save_partial_img_path)



def get_model_path(bench_params: BenchmarkParams):
    return '{0}/weights-{1}-{2}-{3}.hdf5'.format(bench_params.save_model_path, bench_params.id,
                                                    bench_params.curr_sym,
                                                    bench_params.curr_iter_num)


def copy_best_and_cleanup_files(bench_params: BenchmarkParams, max_index, rounded_acc):
    sym = bench_params.curr_sym
    prev_iter_num = bench_params.curr_iter_num
    bench_params.curr_iter_num = max_index + 1
    id = bench_params.id
    model_path = bench_params.save_model_path
    if bench_params.save_files:
        copyfile(get_model_path(bench_params),
                 '{0}/weights-{1}-{2}-{3}.hdf5'.format(model_path, id, sym, rounded_acc))
        copyfile('{0}/{1}-{2}-{3}.png'.format(bench_params.save_img_path, id, sym, max_index + 1),
                 '{0}/{1}-{2}-{3}.png'.format(bench_params.save_img_path, id, sym, rounded_acc))
        copyfile('{0}/{1}-{2}-{3}.pdf'.format(bench_params.save_img_path, id, sym, max_index + 1),
                 '{0}/{1}-{2}-{3}.pdf'.format(bench_params.save_img_path, id, sym, rounded_acc))

        if bench_params.cleanup_files:
            for f in os.listdir(model_path):
                if re.search('weights-{0}-{1}-\d+\.hdf5'.format(id, sym), f):
                    os.remove(os.path.join(model_path, f))

            for f in os.listdir(bench_params.save_img_path):
                if re.search('{0}-{1}-\d+\.(png|pdf)'.format(id, sym), f):
                    os.remove(os.path.join(bench_params.save_img_path, f))

    bench_params.curr_iter_num = prev_iter_num


def get_img_path(bench_params: BenchmarkParams):
    return '{0}-{1}-{2}'.format(bench_params.id, bench_params.curr_sym, bench_params.curr_iter_num)


def save_results(results_df, bench_params: BenchmarkParams):
    if results_df is not None and len(results_df) > 0:
        results_df.to_csv('{0}/results-{1}.csv'.format(bench_params.benchmark_path, bench_params.benchmark_name),
                          index=False)


def get_patrial_img_path(bench_params, name):
    return '{0}-{1}-{2}-{3}'.format(bench_params.id, bench_params.curr_sym, name, bench_params.curr_iter_num)

