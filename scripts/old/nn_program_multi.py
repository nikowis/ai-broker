import time

from data_import import api_to_db_importer
import csv_importer
from old import nn_model, data_helper
import plot_helper
import stock_constants as const

VERBOSE = 2

TARGET_DIR = './../target'
CSV_FILES_DIR = TARGET_DIR + '/data'
DAYS_IN_WINDOW = 5
STOCK_COMPANIES = 10
BINARY_CLASSIFICATION = True
ACTIVATION = 'relu'
OPTIMIZER = 'Adam'
LOSS_FUN = 'categorical_crossentropy'
EPOCHS = 50
BATCH_SIZE = 10
LAYERS = [10, 10, 10]

COLUMNS = [const.VOLUME_COL, const.OPEN_COL, const.ADJUSTED_CLOSE_COL, const.HIGH_COL, const.LOW_COL,
           const.HL_PCT_CHANGE_COL]


def main():

    model_filepath = TARGET_DIR +'/model.days' + str(
        DAYS_IN_WINDOW) + '.epochs{epoch:02d}-accuracy{val_categorical_accuracy:.3f}.hdf5'
    total_time = time.time()
    callbacks = [
        #EarlyStopping(monitor='val_loss', min_delta=0.005, patience=20, verbose=0, mode='auto'),
        # ModelCheckpoint(model_filepath, monitor='val_categorical_accuracy', verbose=0, save_best_only=True, mode='auto',
        #                 period=5)
    ]

    symbols = api_to_db_importer.SYMBOLS[0:STOCK_COMPANIES]
    df_list = csv_importer.import_data_from_files(symbols, CSV_FILES_DIR)

    x_train, x_test, y_train_one_hot, y_test_one_hot = data_helper.extract_data_from_list(df_list, 0,
                                                                                          binary_classification=BINARY_CLASSIFICATION)

    file_name = get_report_file_name(ACTIVATION, DAYS_IN_WINDOW, LOSS_FUN, OPTIMIZER)

    print('\nSTARTING TRAINING FOR ' + file_name)

    _, classes_count = y_test_one_hot.shape
    model = nn_model.create_seq_model(LAYERS, input_size=x_train.shape[1], activation=ACTIVATION,
                                      optimizer=OPTIMIZER,
                                      loss=LOSS_FUN, class_count=classes_count)

    history = model.fit(x_train, y_train_one_hot, validation_data=(x_test, y_test_one_hot),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE, verbose=VERBOSE, callbacks=callbacks)
    loss, accuracy = model.evaluate(x_test, y_test_one_hot, verbose=0)
    print("Loss: ", loss, " Accuracy: ", accuracy, " EPOCHS: ", EPOCHS)

    main_title = get_report_title(accuracy, ACTIVATION, EPOCHS, DAYS_IN_WINDOW, LAYERS, loss, LOSS_FUN, OPTIMIZER)
    y_test_score = model.predict(x_test)

    plot_helper.plot_result(y_test_one_hot, y_test_score, classes_count, history, main_title,
                            file_name,target_dir=TARGET_DIR)

    print('Total time ', str(int(time.time() - total_time)), 's')


def get_report_title(accuracy, ACTIVATION, EPOCHS, DAYS_IN_WINDOW, LAYERS, loss, LOSS_FUN, OPTIMIZER):
    main_title = "Loss: " + str(round(loss, 4)) + ", accuracy: " + str(
        round(accuracy, 4)) + ", EPOCHS: " + str(
        EPOCHS) + ', history days:' + str(DAYS_IN_WINDOW) + '\n'
    main_title += 'Layers: [' + ''.join(
        str(e) + " " for e in LAYERS) + '], optimizer: ' + str(OPTIMIZER) + ', loss: ' + str(
        LOSS_FUN) + ', activation: ' + str(
        ACTIVATION)
    return main_title


def get_report_file_name(ACTIVATION, DAYS_IN_WINDOW, LOSS_FUN, OPTIMIZER):
    return str(DAYS_IN_WINDOW) + 'LOSS_' + str(LOSS_FUN) + '_ACTIVATION_' + str(
        ACTIVATION) + '_OPTIMIZER_' + str(
        OPTIMIZER) + '_HIST_' + str(DAYS_IN_WINDOW)


if __name__ == '__main__':
    main()
