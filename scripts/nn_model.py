from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2


def create_seq_model(hidden_layer_neuron_arr, activation=None, optimizer='adam', loss='categorical_crossentropy',
                     metric='categorical_accuracy', use_bias=True, output_neurons=3, input_size=1, overfitting_regularizer=0.01):
    model = Sequential()
    regularizer = l2(overfitting_regularizer)
    model.add(Dense(hidden_layer_neuron_arr[0], input_shape=(input_size,), activation=activation, use_bias=use_bias,
                    kernel_regularizer=regularizer, bias_regularizer=regularizer))
    if len(hidden_layer_neuron_arr) > 1:
        for layer in hidden_layer_neuron_arr[1:]:
            model.add(Dense(layer, activation=activation, use_bias=use_bias, kernel_regularizer=regularizer, bias_regularizer=regularizer))

    model.add(Dense(output_neurons, activation='sigmoid'))
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[metric])

    return model
