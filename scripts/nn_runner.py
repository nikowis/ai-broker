import db_access
import data_helper
import plot_helper
import nn_model

LAYERS = [20, 20, 20]
EPOCHS = 5
ACTIVATION = 'relu'
OPTIMIZER = 'adam'
LOSS_FUN = 'mean_squared_error'
BATCH_SIZE = 10
TICKER = 'DXYN'
HISTORY_DAYS = 1


def run(x_train, x_test, y_train_one_hot, y_test_one_hot, layers=LAYERS, epochs=EPOCHS,
        activation=ACTIVATION, optimizer=OPTIMIZER,
        loss_fun=LOSS_FUN, batch_size=BATCH_SIZE, history_days=HISTORY_DAYS, file_name='model_stats',
        outstanding_treshold=0.42):
    _, classes_count = y_test_one_hot.shape
    model = nn_model.create_seq_model(layers, input_size=x_train.shape[1], activation=activation, optimizer=optimizer,
                                      loss=loss_fun, class_count=classes_count)

    history = model.fit(x_train, y_train_one_hot, validation_data=(x_test, y_test_one_hot), epochs=epochs,
                        batch_size=batch_size, verbose=0)
    loss, accuracy = model.evaluate(x_test, y_test_one_hot, verbose=0)
    print("Loss: ", loss, " Accuracy: ", accuracy, " epochs: ", epochs)

    main_title = "Loss: " + str(round(loss, 4)) + ", accuracy: " + str(round(accuracy, 4)) + ", epochs: " + str(
        epochs) + ', history days:' + str(history_days) + '\n'
    main_title += 'Layers: [' + ''.join(
        str(e) + " " for e in layers) + '], optimizer: ' + optimizer + ', loss: ' + str(
        loss_fun) + ', activation: ' + str(
        activation)
    y_test_score = model.predict(x_test)

    plot_helper.plot_result(y_test_one_hot, y_test_score, classes_count, history, main_title, file_name,
                            accuracy >= outstanding_treshold)


if __name__ == '__main__':
    db_conn = db_access.create_db_connection()
    df = db_access.find_by_tickers_to_dateframe_parse_to_df_list(db_conn, [TICKER])[0][0]
    df_modified, x_standardized, x_train, x_test, y_train_one_hot, y_test_one_hot = data_helper.extract_data(df, HISTORY_DAYS)

    run(x_train, x_test, y_train_one_hot, y_test_one_hot)
