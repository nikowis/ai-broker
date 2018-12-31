import time

import data_helper
import db_access
import nn_runner

MIN_DATE = '2009-01-01'

def main():
    ticker = 'ACUR'
    db_conn = db_access.create_db_connection(remote=False)
    df_list, sym_list = db_access.find_by_tickers_to_dateframe_parse_to_df_list(db_conn, [ticker], min_date=MIN_DATE)
    df = df_list[0]
    epochs = 30
    # layers = [7, 7, 7]
    skip_iterations = 0

    # 'mean_squared_error', 'logcosh', 'categorical_crossentropy', 'binary_crossentropy'
    losses = ['binary_crossentropy']

    # 'relu, 'softmax'
    activations = ['relu']

    # 'sgd', 'adam', 'rmsprop'
    optimizers = ['adam']

    total_time = time.time()
    iteration = 0
    for hist_dayz in range(0, 10, 1):
        df_modified, x_standardized, x_train, x_test, y_train_one_hot, y_test_one_hot = data_helper.extract_data(df,
                                                                                                                 hist_dayz,
                                                                                                                 binary_classification=True)
        for optmzr in optimizers:
            for actv in activations:
                for lss in losses:
                    iteration += 1
                    if iteration > skip_iterations:
                        file_name = get_report_file_name(actv, hist_dayz, iteration, lss, optmzr)
                        print('\nSTARTING TRAINING FOR ' + file_name)
                        neuron_count = x_train.shape[1] - 1
                        layers = [neuron_count, neuron_count, neuron_count]
                        iter_time = time.time()
                        nn_runner.run(x_train, x_test, y_train_one_hot, y_test_one_hot,
                                      epochs=epochs,
                                      layers=layers, optimizer=optmzr, loss_fun=lss, activation=actv,
                                      history_days=hist_dayz, file_name=file_name, outstanding_treshold=0.6)
                        print('Total time ', str(int(time.time() - total_time)),
                              's, iteration ' + str(iteration) + ' time ', str(int(time.time() - iter_time)), 's.')


def get_report_file_name(actv, hist_dayz, iteration, lss, optmzr):
    return str(iteration) + '_LOSS_' + str(lss) + '_ACTIVATION_' + str(
        actv) + '_OPTIMIZER_' + str(
        optmzr) + '_HIST_' + str(hist_dayz)


if __name__ == '__main__':
    main()
