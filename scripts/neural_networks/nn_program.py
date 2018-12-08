from db import db_access
from helpers import data_helper
from neural_networks import nn_runner
import time


def main():
    ticker = 'CALM'
    db_conn = db_access.create_db_connection(remote=False)
    df = db_access.find_one_by_ticker_dateframe(db_conn, ticker)

    epochs = 200
    layers = [10,10,10]
    skip_iterations = 0

    losses = ['mean_squared_error',
              # 'categorical_crossentropy'
              ]

    activations = ['relu']

    # adam tends to overfit
    optimizers = [#'adam',
                  'sgd']
    total_time = time.time()
    iteration = 0
    for hist_dayz in range(0, 50, 1):
        df, x_standarized, x_train, x_test, y_train_binary, y_test_binary = data_helper.extract_data(df, 1, hist_dayz)
        for optmzr in optimizers:
            for actv in activations:
                for lss in losses:
                    iteration += 1
                    if iteration > skip_iterations:
                        file_name = get_report_file_name(actv, hist_dayz, iteration, lss, optmzr)
                        print('\nSTARTING TRAINING FOR ' + file_name)
                        iter_time = time.time()
                        nn_runner.run(df, x_standarized, x_train, x_test, y_train_binary, y_test_binary,
                                      epochs=epochs,
                                      layers=layers, optimizer=optmzr, loss_fun=lss, activation=actv,
                                      history_days=hist_dayz, file_name=file_name, outstanding_treshold=0.40)
                        print('Total time ', str(int(time.time()-total_time)), 's, iteration '+str(iteration)+' time ', str(int(time.time() - iter_time)), 's.')


def get_report_file_name(actv, hist_dayz, iteration, lss, optmzr):
    return str(iteration) + '_LOSS_' + str(lss) + '_ACTIVATION_' + str(
        actv) + '_OPTIMIZER_' + str(
        optmzr) + '_HIST_' + str(hist_dayz)


if __name__ == '__main__':
    main()
