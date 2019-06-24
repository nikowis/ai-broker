
from benchmark_params import LightGBMBenchmarkParams
from benchmark import Benchmark
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score

class LightGBMBenchmark(Benchmark):

    def __init__(self, symbols, bench_params: LightGBMBenchmarkParams, changing_params_dict: dict = None) -> None:
        super().__init__(symbols, bench_params, changing_params_dict)

    def create_model(self, bench_params: LightGBMBenchmarkParams):
        return None

    def create_callbacks(self, bench_params):
        """Create callbacks used while learning"""
        pass

    def get_walk_forward_epochs(self, bench_params, iteration):
        """Get walk forward epochs for iteration"""
        return None


    def evaluate_predict(self, model, x_test, y_test):
        """Evaluate on test data, predict labels for x_test, return (accuracy, loss, y_prediction)"""
        y_test_prediction = model.predict(x_test)
        y_test_prediction_parsed = np.array([np.argmax(pred, axis=None, out=None) for pred in y_test_prediction])
        acc = accuracy_score(y_test, y_test_prediction_parsed)
        return acc, 0, y_test_prediction

    def fit_model(self, bench_params:LightGBMBenchmarkParams, model, callbacks, x_train, y_train, x_test, y_test, epochs=None):
        """Fit model on train data, return learning history or none"""
        params = {
            "objective": "multiclass",
            "num_class": bench_params.classes_count,
            "num_leaves": 60,
            "max_depth": -1,
            "learning_rate": 0.01,
            "bagging_fraction": 0.9,  # subsample
            "feature_fraction": 0.9,  # colsample_bytree
            "bagging_freq": 5,  # subsample_freq
            "bagging_seed": 2018,
            "verbosity": -1}

        train_data = lgb.Dataset(x_train, label=y_train)
        test_data = lgb.Dataset(x_test, label=y_test)
        bst = lgb.train(params, train_data, num_boost_round=10, valid_sets=[test_data])
        return bst, None

    def update_walk_history(self, bench_params, history, walk_history):
        """Update history object with walk forward learning history"""
        pass

    def create_history_object(self, bench_params):
        """Create an empty history object for walk forward learning"""
        pass


if __name__ == '__main__':
    bench_params = LightGBMBenchmarkParams(False, benchmark_name='lgbm_first')
    LightGBMBenchmark(['GOOGL'], bench_params)
