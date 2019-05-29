from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2


def create_seq_model(hidden_layer_neuron_arr, activation=None, optimizer='adam', loss='categorical_crossentropy',
                     metric='categorical_accuracy', use_bias=True, output_neurons=3, input_size=1):
    model = Sequential()
    model.add(Dense(hidden_layer_neuron_arr[0], input_shape=(input_size,), activation=activation, use_bias=use_bias,
                    kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    if len(hidden_layer_neuron_arr) > 1:
        for layer in hidden_layer_neuron_arr[1:]:
            model.add(Dense(layer, activation=activation, use_bias=use_bias, kernel_regularizer=l2(0.01),
                            bias_regularizer=l2(0.01)))

    model.add(Dense(output_neurons, activation='sigmoid', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[metric])

    return model
