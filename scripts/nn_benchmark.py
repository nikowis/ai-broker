import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint

import benchmark_file_helper
import benchmark_nn_model
import benchmark_params
from benchmark import Benchmark
from benchmark_params import NnBenchmarkParams


class NnBenchmark(Benchmark):
    def __init__(self, symbols, bench_params: NnBenchmarkParams, changing_params_dict: dict = None) -> None:
        super().__init__(symbols, bench_params, changing_params_dict)

    def create_callbacks(self):
        bench_params = self.bench_params
        earlyStopping = EarlyStopping(monitor='val_' + bench_params.metric,
                                      min_delta=bench_params.early_stopping_min_delta,
                                      patience=bench_params.early_stopping_patience, verbose=0, mode='max',
                                      restore_best_weights=True)
        callbacks = [earlyStopping]
        if bench_params.save_files:
            mcp_save = ModelCheckpoint(
                benchmark_file_helper.get_model_path(bench_params), save_best_only=True,
                monitor='val_' + bench_params.metric, mode='max')
            callbacks = [earlyStopping, mcp_save]
        return callbacks

    def create_model(self):
        bench_params = self.bench_params
        return benchmark_nn_model.create_seq_model(bench_params)

    def evaluate_predict(self, model, x_test, y_test):
        ls, acc = model.evaluate(x_test, y_test, verbose=0)
        y_test_prediction = model.predict(x_test)
        return acc, ls, y_test_prediction

    def fit_model(self, model, callbacks, x_train, y_train, x_test, y_test):
        bench_params = self.bench_params
        epochs = bench_params.epochs
        return model, model.fit(x_train, y_train, validation_data=(x_test, y_test),
                         epochs=epochs, batch_size=bench_params.batch_size,
                         callbacks=callbacks, verbose=0)

    def update_walk_history(self, history, walk_history):
        bench_params = self.bench_params
        walk_history.history['loss'] += history.history['loss']
        walk_history.history['val_loss'] += history.history['val_loss']
        walk_history.history[bench_params.metric] += history.history[bench_params.metric]
        walk_history.history['val_' + bench_params.metric] += history.history[
            'val_' + bench_params.metric]

    def create_history_object(self):
        bench_params = self.bench_params
        walk_history = keras.callbacks.History()
        walk_history.history = {'loss': [], 'val_loss': [], bench_params.metric: [],
                                'val_' + bench_params.metric: []}
        return walk_history


if __name__ == '__main__':

    bench_params = benchmark_params.NnBenchmarkParams(
        binary_classification=True
        , examined_param='pca'
        , benchmark_name='nn-benchmark-pca'

    )
    bench_params.iterations = 5
    bench_params.walk_forward_testing = True
    bench_params.walk_forward_test_window_size = 180
    bench_params.epochs = 10
    NnBenchmark(
        ['GOOGL']
        , bench_params
        , changing_params_dict={'pca': [None, 0.9999, 0.999, 0.99]}
    )
