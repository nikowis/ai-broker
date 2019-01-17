import time

import data_helper
import db_access
import nn_model
import plot_helper

MIN_DATE = '2009-01-01'

def main():
    ticker = 'ACUR'
    db_conn = db_access.create_db_connection(remote=False)
    df_list, sym_list = db_access.find_by_tickers_to_dateframe_parse_to_df_list(db_conn, [ticker], min_date=MIN_DATE)
    df = df_list[0]
    epochs = 50
    batch_size=10
    layers = [4, 4, 4]
    skip_iterations = 0

    # 'mean_squared_error', 'logcosh', 'categorical_crossentropy', 'binary_crossentropy'
    losses = ['binary_crossentropy']

    # 'relu, 'softmax'
    activations = ['relu']

    # 'sgd', 'adam', 'rmsprop'
    optimizers = ['adam']

    total_time = time.time()
    iteration = 0
    for hist_dayz in range(0, 3, 1):
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
                        neuron_count = int(x_train.shape[1]/2)+1
                        layers = [neuron_count, neuron_count, neuron_count]
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
                                                accuracy >= 0.6)

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
