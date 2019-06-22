from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2

from benchmark_params import NnBenchmarkParams


def create_seq_model(bench_params:NnBenchmarkParams):
    model = Sequential()
    layers = bench_params.layers

    if bench_params.regularizer is not None:
        regularizer = l2(bench_params.regularizer)
    else:
        regularizer = None
    if len(layers) == 0:
        model.add(Dense(bench_params.output_neurons, use_bias=bench_params.use_bias, input_shape=(bench_params.input_size,),
                        activation=bench_params.output_activation))
    else:
        model.add(
            Dense(layers[0], input_shape=(bench_params.input_size,), activation=bench_params.activation,
                  use_bias=bench_params.use_bias,
                  kernel_regularizer=regularizer, bias_regularizer=regularizer))
        if len(layers) > 1:
            for layer in layers[1:]:
                model.add(Dense(layer, activation=bench_params.activation, use_bias=bench_params.use_bias,
                                kernel_regularizer=regularizer, bias_regularizer=regularizer))

        model.add(Dense(bench_params.output_neurons, activation=bench_params.output_activation,
                        use_bias=bench_params.use_bias))

    model.compile(optimizer=bench_params.optimizer,
                  loss=bench_params.loss,
                  metrics=[bench_params.metric])

    return model
