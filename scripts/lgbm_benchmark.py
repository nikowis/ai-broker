import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from benchmark import Benchmark
from benchmark_params import LightGBMBenchmarkParams


class LightGBMBenchmark(Benchmark):

    def __init__(self, symbols, bench_params: LightGBMBenchmarkParams, changing_params_dict: dict = None) -> None:
        super().__init__(symbols, bench_params, changing_params_dict)

    def create_model(self):
        return None

    def create_callbacks(self):
        """Create callbacks used while learning"""
        pass

    def get_walk_forward_epochs(self, iteration):
        """Get walk forward epochs for iteration"""
        return None

    def evaluate_predict(self, model, x_test, y_test):
        """Evaluate on test data, predict labels for x_test, return (accuracy, loss, y_prediction)"""
        y_test_prediction = model.predict(x_test, num_iteration=model.best_iteration)
        if self.bench_params.binary_classification:
            y_test_prediction_parsed = np.array(y_test_prediction, copy=True)
            y_test_prediction_parsed[y_test_prediction >= 0.5] = 1
            y_test_prediction_parsed[y_test_prediction < 0.5] = 0
        else:
            y_test_prediction_parsed = np.array([np.argmax(pred, axis=None, out=None) for pred in y_test_prediction])
        acc = accuracy_score(y_test, y_test_prediction_parsed)
        return acc, 0, y_test_prediction

    def fit_model(self, model, callbacks, x_train, y_train, x_test, y_test,
                  epochs=None):
        """Fit model on train data, return learning history or none"""
        bench_params = self.bench_params
        params = {
            "objective": bench_params.objective,
            "num_class": bench_params.model_num_class,
            "num_leaves": bench_params.num_leaves,
            "max_depth": bench_params.max_depth,
            "learning_rate": bench_params.learning_rate,
            "boosting": bench_params.boosting,
            "num_threads": 2,
            "max_bin": bench_params.max_bin,
            # "bagging_fraction" : bench_params.bagging_fraction,
            # "bagging_freq" : bench_params.bagging_freq,
            "feature_fraction": bench_params.feature_fraction,
            "verbosity": -1
        }

        evaluating_history = {}

        train_data = lgb.Dataset(x_train, label=y_train)
        test_data = lgb.Dataset(x_test, label=y_test)
        bst = lgb.train(params, train_data, valid_sets=[test_data, train_data], num_boost_round=bench_params.num_boost_round
                        , early_stopping_rounds=20, verbose_eval=20)#, evals_result=evaluating_history)

        # print('Plotting metrics recorded during training...')
        # ax = lgb.plot_metric(evaluating_history, metric='l1')
        # plt.show()
        #
        # print('Plotting feature importances...')
        # ax = lgb.plot_importance(bst, max_num_features=10)
        # plt.show()
        #
        # print('Plotting split value histogram...')
        # ax = lgb.plot_split_value_histogram(bst, feature='f26', bins='auto')
        # plt.show()
        #
        # print('Plotting 54th tree...')  # one tree use categorical feature to split
        # ax = lgb.plot_tree(bst, tree_index=53, figsize=(15, 15), show_info=['split_gain'])
        # plt.show()

        return bst, None

    def update_walk_history(self, history, walk_history):
        """Update history object with walk forward learning history"""
        pass

    def create_history_object(self):
        """Create an empty history object for walk forward learning"""
        pass


if __name__ == '__main__':
    bench_params = LightGBMBenchmarkParams(False, benchmark_name='lgbm_first')
    LightGBMBenchmark(['GOOGL'], bench_params)
