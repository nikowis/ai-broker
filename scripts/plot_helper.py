import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style
from sklearn.metrics import roc_curve, auc
from keras.utils import to_categorical


import stock_constants as const

BASE_IMG_PATH = './../target'



CLOSE_PRICE_USD_LABEL = 'Cena zamknięcia (USD)'
DATE_LABEL = 'Data'
FORECAST_LABEL = 'Prognoza'
CLOSE_PRICE_LABEL = 'Cena zamknięcia'
PRICE_CHANGE_PCT_LABEL = 'Zmiana ceny (%)'
FORECAST_COUNT_LABEL = 'Liczba prognoz'
VALUE_CHANGE_LABEL = 'Zmiana wartości'
RATE_CHANGE_FORECAST_LABEL = 'Przewidywana zmiana kursu'
RATE_CHANGE_LABEL = 'Zmiana kursu'
RISE_LABEL = 'wzrost'
IDLE_LABEL = 'utrzymanie'
FALL_LABEL = 'spadek'
PREDICTED_LEGEND = 'Przewidywane'
REAL_LEGEND = 'Rzeczywiste'

HISTOGRAM_TITLE = 'Histogram predykcji dla zbioru testowego'
EPOCH_LABEL = 'Epoka'
ACCURACY_LABEL = 'dokładność (%)'
LOSS_LABEL = 'wartość funkcji straty'
LOSS_TITLE = 'Strata'
ACCURACY_TITLE = 'Dokładność'
TEST_DATA = 'Dane testowe'
TRAIN_DATA = 'Dane treningowe'

# ROC
FPR_LABEL = 'False Positive Rate'
TPR_LABEL = 'True Positive Rate'
ROC_TITLE = 'Krzywe ROC'
MICRO_ROC_KEY = "micro"
CLASS_ROC_LABEL = "Klasa '{0}' (obszar {1:0.2f})"
MICRO_AVG_ROC_LABEL = 'Mikro-średnia klas (obszar {0:0.2f})'
BINARY_ROC_LABEL = 'Krzywa ROC (obszar {0:0.2f})'


def legend_labels_save_files(title, file_name='img', base_img_path=BASE_IMG_PATH, xlabel=DATE_LABEL,
                             ylabel=CLOSE_PRICE_USD_LABEL, legend=4):
    if not legend == -1:
        plt.legend(loc=legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig('{}/{}.pdf'.format(base_img_path, file_name), format='pdf', dpi=1000)
    plt.savefig('{}/{}.png'.format(base_img_path, file_name))
    plt.show()
    plt.close()


def calculate_roc_auc(y_test, y_test_score, classes_count):
    if classes_count>2:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(classes_count):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_test_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr[MICRO_ROC_KEY], tpr[MICRO_ROC_KEY], _ = roc_curve(y_test.ravel(), y_test_score.ravel())
        roc_auc[MICRO_ROC_KEY] = auc(fpr[MICRO_ROC_KEY], tpr[MICRO_ROC_KEY])

        return fpr, tpr, roc_auc
    else:
        fpr, tpr, _ = roc_curve(y_test, y_test_score)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc


def plot_result(y_test, y_test_prediction, classes_count, history, main_title, file_name, target_dir=BASE_IMG_PATH):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if classes_count == 2:
        class_labels = [FALL_LABEL, RISE_LABEL]
        xticks = [0, 1]
        y_test_prediction = y_test_prediction.flatten()
        fpr, tpr, roc_auc = calculate_roc_auc(y_test, y_test_prediction, classes_count)
        y_test_prediction[y_test_prediction >= 0.5] = 1
        y_test_prediction[y_test_prediction < 0.5] = 0
        y_test_prediction = to_categorical(y_test_prediction)
        y_test = to_categorical(y_test)
    else:
        class_labels = [FALL_LABEL, IDLE_LABEL, RISE_LABEL]
        xticks = [0, 1, 2]
        fpr, tpr, roc_auc = calculate_roc_auc(y_test, y_test_prediction, classes_count)

    y_test = [np.argmax(pred, axis=None, out=None) for pred in y_test]
    y_test_prediction = [np.argmax(pred, axis=None, out=None) for pred in y_test_prediction]

    plt.figure(figsize=(12, 12))
    style.use('ggplot')
    plt.suptitle(main_title)

    plt.subplot(2, 2, 1)


    dftmp = pd.DataFrame({'tmpcol': y_test})

    dftmp['tmpcol'].plot(kind='hist', xticks=xticks, alpha=0.7, label=RATE_CHANGE_LABEL)

    dftmp = pd.DataFrame.from_records({'tmpcol': y_test_prediction})

    dftmp['tmpcol'].plot(kind='hist', xticks=xticks, alpha=0.7, label=RATE_CHANGE_LABEL)

    plt.legend([REAL_LEGEND, PREDICTED_LEGEND], loc='upper left')
    plt.xticks(xticks, class_labels)
    plt.xlabel(VALUE_CHANGE_LABEL)
    plt.ylabel(FORECAST_COUNT_LABEL)
    plt.title(HISTOGRAM_TITLE)

    plt.subplot(2, 2, 2)
    if classes_count>2:
        plt.plot(fpr[(MICRO_ROC_KEY)], tpr[MICRO_ROC_KEY],
                 label=MICRO_AVG_ROC_LABEL.format(roc_auc[MICRO_ROC_KEY]),
                 color='red', linewidth=3)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(classes_count), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=CLASS_ROC_LABEL.format(class_labels[i], roc_auc[i]))
    else:
        plt.plot(fpr, tpr, label=BINARY_ROC_LABEL.format(roc_auc),color='red', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(FPR_LABEL)
    plt.ylabel(TPR_LABEL)
    plt.title(ROC_TITLE)
    plt.legend(loc="lower right")

    if history is not None:
        plt.subplot(2, 2, 3)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(LOSS_TITLE)
        plt.ylabel(LOSS_LABEL)
        plt.xlabel(EPOCH_LABEL)
        plt.legend([TRAIN_DATA, TEST_DATA], loc='upper left')

        plt.subplot(2, 2, 4)
        if classes_count > 2:
            plt.plot(history.history['categorical_accuracy'])
            plt.plot(history.history['val_categorical_accuracy'])
        else:
            plt.plot(history.history['binary_accuracy'])
            plt.plot(history.history['val_binary_accuracy'])

        plt.title(ACCURACY_TITLE)
        plt.ylabel(ACCURACY_LABEL)
        plt.xlabel(EPOCH_LABEL)
        plt.legend([TRAIN_DATA, TEST_DATA], loc='upper left')

    # plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # plt.savefig('{}/{}.pdf'.format(target_dir, file_name), format='pdf', dpi=1000)
    plt.savefig('{}/{}.png'.format(target_dir, file_name))
    # plt.show()
    plt.close()


def plot_company_summary(df, symbol):
    plt.figure(figsize=(12, 12))
    style.use('ggplot')
    plt.suptitle(symbol)

    plt.subplot(2, 2, 1)
    df[const.ADJUSTED_CLOSE_COL].plot(kind='line')
    plt.title('Cena zamknięcia')
    plt.ylabel('cena')
    plt.xlabel('data')

    plt.subplot(2, 2, 2)
    df[const.DAILY_PCT_CHANGE_COL].plot(kind='line')
    plt.title('Zmiana % względem dnia następnego')
    plt.ylabel('%')
    plt.xlabel('data')

    plt.subplot(2, 2, 3)
    df[const.HL_PCT_CHANGE_COL] = (df[const.HIGH_COL] - df[const.LOW_COL]) / df[
        const.HIGH_COL] * 100
    df[const.HL_PCT_CHANGE_COL].plot(kind='line')
    plt.title('Procentowa zmiana H/L')
    plt.ylabel('%')
    plt.xlabel('data')

    plt.subplot(2, 2, 4)
    df[const.LABEL_DISCRETE_COL].plot(kind='hist', xticks=[0, 1, 2], label=RATE_CHANGE_LABEL)
    plt.xticks([0, 1, 2], [FALL_LABEL, IDLE_LABEL ,RISE_LABEL])
    plt.xlabel(VALUE_CHANGE_LABEL)
    plt.ylabel(FORECAST_COUNT_LABEL)
    plt.title(HISTOGRAM_TITLE)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    COMPANY_INFO_PATH = BASE_IMG_PATH + '/company_info'
    if not os.path.exists(COMPANY_INFO_PATH):
        os.makedirs(COMPANY_INFO_PATH)
    plt.savefig('{}/{}.png'.format(COMPANY_INFO_PATH, symbol))

    # plt.show()
    plt.close()
