from sklearn.ensemble import RandomForestClassifier

import benchmark_params
from benchmark import Benchmark
from benchmark_params import RandomForestBenchmarkParams


class RandomForestBenchmark(Benchmark):

    def __init__(self, symbols, bench_params: RandomForestBenchmarkParams, changing_params_dict: dict = None) -> None:
        super().__init__(symbols, bench_params, changing_params_dict)

    def create_model(self):
        bench_params = self.bench_params
        return RandomForestClassifier(n_estimators=bench_params.n_estimators, criterion=bench_params.criterion
                                      , max_depth=bench_params.max_depth,
                                      min_samples_split=bench_params.min_samples_split
                                      , min_samples_leaf=bench_params.min_samples_leaf
                                      , min_weight_fraction_leaf=bench_params.min_weight_fraction_leaf
                                      , max_features=bench_params.max_features
                                      , max_leaf_nodes=bench_params.max_leaf_nodes
                                      , min_impurity_decrease=bench_params.min_impurity_decrease
                                      , bootstrap=bench_params.bootstrap
                                      , warm_start=bench_params.warm_start
                                      , oob_score=bench_params.oob_score)

    def create_callbacks(self):
        """Create callbacks used while learning"""
        pass

    def evaluate_predict(self, model, x_test, y_test):
        """Evaluate on test data, predict labels for x_test, return (accuracy, loss, y_prediction)"""
        acc = model.score(x_test, y_test)
        y_test_prediction = model.predict_proba(x_test)
        return acc, 0, y_test_prediction

    def fit_model(self, model, callbacks, x_train, y_train, x_test, y_test, epochs=None):
        """Fit model on train data, return learning history or none"""

        model.fit(x_train, y_train)
        return model, None

    def update_walk_history(self, history, walk_history):
        """Update history object with walk forward learning history"""
        pass

    def create_history_object(self):
        """Create an empty history object for walk forward learning"""
        pass


if __name__ == '__main__':
    bench_params = benchmark_params.RandomForestBenchmarkParams(True, benchmark_name='rf_first')
    RandomForestBenchmark(['GOOGL'], bench_params)
