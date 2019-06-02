from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2


def create_seq_model(input_size, model_params):
    model = Sequential()
    layers = model_params.layers

    regularizer = l2(model_params.regularizer)
    model.add(
        Dense(layers[0], input_shape=(input_size,), activation=model_params.activation, use_bias=model_params.use_bias,
              kernel_regularizer=regularizer, bias_regularizer=regularizer))
    if len(layers) > 1:
        for layer in layers[1:]:
            model.add(Dense(layer, activation=model_params.activation, use_bias=model_params.use_bias,
                            kernel_regularizer=regularizer, bias_regularizer=regularizer))

    model.add(Dense(model_params.output_neurons, activation=model_params.output_activation))
    model.compile(optimizer=model_params.optimizer,
                  loss=model_params.loss,
                  metrics=[model_params.metric])

    return model

