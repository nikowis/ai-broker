from sklearn.svm import SVC

import benchmark_params
from benchmark import Benchmark
from benchmark_params import SVMBenchmarkParams


class SVMBenchmark(Benchmark):

    def __init__(self, symbols, bench_params: SVMBenchmarkParams, changing_params_dict: dict = None) -> None:
        super().__init__(symbols, bench_params, changing_params_dict)

    def create_model(self):
        bench_params = self.bench_params
        return SVC(C=bench_params.c, kernel=bench_params.kernel, degree=bench_params.degree, gamma=bench_params.gamma,
                   probability=True)

    def create_callbacks(self):
        """Create callbacks used while learning"""
        pass

    def get_walk_forward_epochs(self, iteration):
        """Get walk forward epochs for iteration"""
        return None


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
    bench_params = benchmark_params.SVMBenchmarkParams(False, benchmark_name='svm_first')
    SVMBenchmark(['GOOGL'], bench_params)
