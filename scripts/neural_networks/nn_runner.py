import numpy as np

import db.stock_constants as const
from helpers import plot_helper
from neural_networks import nn_model

LAYERS = [20, 20, 20]
EPOCHS = 5
FORECAST_DAYS = 1
ACTIVATION = None
OPTIMIZER = 'adam'
LOSS = 'categorical_crossentropy'
BATCH_SIZE = 10
TICKER = 'GOOGL'
HISTORY_DAYS = 20


def run(df, x_standarized, x_train, x_test, y_train_binary, y_test_binary, layers=LAYERS, epochs=EPOCHS, activation=ACTIVATION, optimizer=OPTIMIZER,
        loss=LOSS, batch_size=BATCH_SIZE, ticker=TICKER, history_days=HISTORY_DAYS, file_name='model_stats'):

    model = nn_model.create_seq_model(layers, input_size=x_train.shape[1], activation=activation, optimizer=optimizer,
                                      loss=loss)

    history = model.fit(x_train, y_train_binary, epochs=epochs, batch_size=batch_size)
    loss, accuracy = model.evaluate(x_test, y_test_binary)

    print("Loss: ", loss, " Accuracy: ", accuracy, " epochs: ", epochs)
    predicted_binary = model.predict(x_standarized)
    predicted = [np.argmax(pred, axis=None, out=None) for pred in predicted_binary]
    df[const.FORECAST_DISCRETE_COL] = predicted
    # count_matching = np.count_nonzero(np.where(df[const.FORECAST_DISCRETE_COL] == df[const.LABEL_DISCRETE_COL], 1, 0))
    # print('Accuracy on full data:', count_matching/len(predicted))

    main_title = "Loss: " + str(round(loss, 4)) + ", accuracy: " + str(round(accuracy, 4)) + ", epochs: " + str(
        epochs) + ', history days:' + str(history_days) + '\n'
    main_title += 'Layers: [' + ''.join(
        str(e) + " " for e in layers) + '], optimizer: ' + optimizer + ', loss: ' + str(loss) + ', activation: ' + str(
        activation)

    plot_helper.plot_result(ticker, df, history, main_title, file_name)


if __name__ == "__main__":
    run()
