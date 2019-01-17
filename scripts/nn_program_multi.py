import time

import api_to_db_importer
import data_helper
import db_access
import nn_model
import plot_helper


def main():
    db_conn = db_access.create_db_connection(remote=False, db_name='ai-broker')

    symbols = api_to_db_importer.SYMBOLS[0:10]
    df_list, symbols = db_access.find_by_tickers_to_dateframe_parse_to_df_list(db_conn, symbols)
    batch_size=10
    epochs = 50
    layers = [7,5,3]
    skip_iterations = 0

    losses = ['binary_crossentropy']

    activations = ['relu']

    optimizers = ['adam']

    total_time = time.time()
    iteration = 0
    for hist_dayz in range(0, 1, 1):
        x_train, x_test, y_train_one_hot, y_test_one_hot = data_helper.extract_data_from_list(df_list, hist_dayz,
                                                                                              binary_classification=True)
        for optmzr in optimizers:
            for actv in activations:
                for lss in losses:
                    iteration += 1
                    if iteration > skip_iterations:
                        file_name = get_report_file_name(actv, hist_dayz, iteration, lss, optmzr)
                        neuron_count = x_train.shape[1] - 1
                        #layers = [neuron_count, neuron_count, neuron_count]
                        print('\nSTARTING TRAINING FOR ' + file_name)
                        iter_time = time.time()

                        _, classes_count = y_test_one_hot.shape
                        model = nn_model.create_seq_model(layers, input_size=x_train.shape[1], activation=actv,
                                                          optimizer=optmzr,
                                                          loss=lss, class_count=classes_count)

                        history = model.fit(x_train, y_train_one_hot, validation_data=(x_test, y_test_one_hot),
                                            epochs=epochs,
                                            batch_size=batch_size, verbose=0)
                        loss, accuracy = model.evaluate(x_test, y_test_one_hot, verbose=0)
                        print("Loss: ", loss, " Accuracy: ", accuracy, " epochs: ", epochs)

                        main_title = get_report_title(accuracy, actv, epochs, hist_dayz, layers, loss, lss, optmzr)
                        y_test_score = model.predict(x_test)

                        plot_helper.plot_result(y_test_one_hot, y_test_score, classes_count, history, main_title,
                                                file_name,
                                                accuracy >= 0.4)

                        print('Total time ', str(int(time.time() - total_time)),
                              's, iteration ' + str(iteration) + ' time ', str(int(time.time() - iter_time)), 's.')


def get_report_title(accuracy, actv, epochs, hist_dayz, layers, loss, lss, optmzr):
    main_title = "Loss: " + str(round(loss, 4)) + ", accuracy: " + str(
        round(accuracy, 4)) + ", epochs: " + str(
        epochs) + ', history days:' + str(hist_dayz) + '\n'
    main_title += 'Layers: [' + ''.join(
        str(e) + " " for e in layers) + '], optimizer: ' + str(optmzr) + ', loss: ' + str(
        lss) + ', activation: ' + str(
        actv)
    return main_title


def get_report_file_name(actv, hist_dayz, iteration, lss, optmzr):
    return str(iteration) + '_LOSS_' + str(lss) + '_ACTIVATION_' + str(
        actv) + '_OPTIMIZER_' + str(
        optmzr) + '_HIST_' + str(hist_dayz)


if __name__ == '__main__':
    main()
