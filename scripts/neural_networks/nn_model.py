from keras.layers import Dense
from keras.models import Sequential


def create_seq_model(hidden_layer_neuron_arr, activation=None, optimizer='adam', loss='categorical_crossentropy',
                     metric='categorical_accuracy', use_bias=True, class_count=3, input_size=1):
    model = Sequential()
    model.add(Dense(hidden_layer_neuron_arr[0], input_shape=(input_size,), activation=activation, use_bias=use_bias))
    if len(hidden_layer_neuron_arr) > 1:
        for layer in hidden_layer_neuron_arr[1:]:
            model.add(Dense(layer, activation=activation, use_bias=use_bias))

    model.add(Dense(class_count))
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[metric])

    return model
