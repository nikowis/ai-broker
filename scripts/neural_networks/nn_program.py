from db import db_access
from helpers import data_helper
from neural_networks import nn_runner


def main():
    ticker = 'GOOGL'
    db_conn = db_access.create_db_connection()
    df = db_access.find_one_by_ticker_dateframe(db_conn, ticker)

    epochs = 100
    layers = [20, 20, 20]
    skip_iterations = 0

    losses = ['mean_squared_error', 'logcosh',
              # 'categorical_crossentropy'
              ]

    activations = ['softmax', 'relu']

    optimizers = ['adam', 'sgd', 'rmsprop']

    iteration = 0
    for hist_dayz in range(6, 20, 1):
        df, x_standarized, x_train, x_test, y_train_binary, y_test_binary = data_helper.extract_data(df, 1, hist_dayz)
        for optmzr in optimizers:
            for actv in activations:
                for lss in losses:
                    iteration += 1
                    if iteration > skip_iterations:
                        if ~is_bad_combination(actv, optmzr):
                            file_name = str(iteration) + '_LOSS_' + str(lss) + '_ACTIVATION_' + str(
                                actv) + '_OPTIMIZER_' + str(
                                optmzr) + '_HIST_' + str(hist_dayz)
                            print('\nSTARTING TRAINING FOR ' + file_name)
                            nn_runner.run(df, x_standarized, x_train, x_test, y_train_binary, y_test_binary,
                                          epochs=epochs,
                                          layers=layers, optimizer=optmzr, loss_fun=lss, activation=actv,
                                          history_days=hist_dayz, file_name=file_name, outstanding_treshold=0.40)


def is_bad_combination(actv, optmzr):
    return (optmzr == 'sgd' and actv == 'softmax')


if __name__ == '__main__':
    main()
