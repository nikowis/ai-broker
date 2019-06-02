import time

import keras

from old import data_helper
from data_import import db_access
import plot_helper
import stock_constants as const
from sklearn.svm import SVC
import numpy as np

MIN_DATE = '1900-01-01'
MAX_DATE = '2020-10-29'

SELECTED_SYM = 'USLM'


def main():
    db_conn = db_access.create_db_connection(remote=False)
    df_list, sym_list = db_access.find_by_tickers_to_dateframe_parse_to_df_list(db_conn, [SELECTED_SYM],
                                                                                min_date=MIN_DATE, max_date=MAX_DATE)
    df = df_list[0]

    COLUMNS = [const.VOLUME_COL, const.OPEN_COL, const.ADJUSTED_CLOSE_COL, const.HIGH_COL, const.LOW_COL,
               const.HL_PCT_CHANGE_COL, const.SMA_5_COL, const.SMA_10_COL, const.SMA_20_COL, const.EMA_12_COL,
               const.EMA_26_COL, const.STOCH_K_COL, const.STOCH_D_COL, const.ROC_10_COL, const.TR_COL, const.MOM_6_COL,
               const.MOM_12_COL, const.WILLR_5_COL, const.WILLR_10_COL, const.APO_6_COL, const.APO_12_COL,
               const.RSI_6_COL, const.RSI_12_COL]


    total_time = time.time()
    iteration = 0
    for hist_dayz in range(0, 1, 1):

        df_modified, x_standardized, x_train, x_test, y_train_one_hot, y_test_one_hot = data_helper.extract_data(df,
                                                                                                                 hist_dayz,
                                                                                                                 binary_classification=True,
                                                                                                                 input_columns=COLUMNS)

        file_name = "svm"
        # print('\nSTARTING TRAINING FOR ' + file_name)

        iter_time = time.time()

        _, classes_count = y_test_one_hot.shape
        model = SVC(C=70, kernel='rbf', gamma=50)
        y = np.array(df[const.LABEL_BINARY_COL])

        y_train = y[0:y_train_one_hot.shape[0]]
        y_test = y[y_train_one_hot.shape[0]+1:]

        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test )
        print("Accuracy: ", accuracy)

        main_title = 'SVM ' + "accuracy: " + str(accuracy)
        y_test_score = model.predict(x_test)
        y_test_score = keras.utils.to_categorical(y_test_score)

        plot_helper.plot_result(y_test_one_hot, y_test_score, classes_count, None, main_title,
                                file_name)

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


if __name__ == '__main__':
    main()
