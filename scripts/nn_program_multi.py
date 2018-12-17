import time

import db_access
import data_helper
import nn_runner


def main():
    db_conn = db_access.create_db_connection(remote=False, db_name='ai-broker')

    symbols = db_access.SELECTED_SYMBOLS_LIST[0:50]
    df_list, symbols = db_access.find_by_tickers_to_dateframe_parse_to_df_list(db_conn, symbols)

    # for i in range(0, len(symbols)):
    #     sym = symbols[i]
    #     df = df_list[i]
    #     plth.plot_company_summary(df, sym)


    epochs = 5
    #layers = [20,20,20]
    skip_iterations = 0

    losses = ['categorical_crossentropy']

    activations = ['relu']

    optimizers = ['adam']

    total_time = time.time()
    iteration = 0
    for hist_dayz in range(1, 5, 1):
        x_train, x_test, y_train_one_hot, y_test_one_hot = data_helper.extract_data_from_list(df_list, hist_dayz)
        for optmzr in optimizers:
            for actv in activations:
                for lss in losses:
                    iteration += 1
                    if iteration > skip_iterations:
                        file_name = get_report_file_name(actv, hist_dayz, iteration, lss, optmzr)
                        neuron_count = x_train.shape[1] - 1
                        layers = [neuron_count, neuron_count, neuron_count]
                        print('\nSTARTING TRAINING FOR ' + file_name)
                        iter_time = time.time()
                        nn_runner.run(x_train, x_test, y_train_one_hot, y_test_one_hot,
                                      epochs=epochs, batch_size=100,
                                      layers=layers, optimizer=optmzr, loss_fun=lss, activation=actv,
                                      history_days=hist_dayz, file_name=file_name, outstanding_treshold=0.40)
                        print('Total time ', str(int(time.time() - total_time)),
                              's, iteration ' + str(iteration) + ' time ', str(int(time.time() - iter_time)), 's.')


def get_report_file_name(actv, hist_dayz, iteration, lss, optmzr):
    return str(iteration) + '_LOSS_' + str(lss) + '_ACTIVATION_' + str(
        actv) + '_OPTIMIZER_' + str(
        optmzr) + '_HIST_' + str(hist_dayz)


if __name__ == '__main__':
    main()
