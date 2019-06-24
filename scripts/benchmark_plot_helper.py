from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from matplotlib import style

import benchmark_file_helper
from benchmark_params import BenchmarkParams
from stock_constants import MICRO_ROC_KEY

CLOSE_PRICE_USD_LABEL = 'Cena zamknięcia (USD)'
DATE_LABEL = 'Data'
FORECAST_LABEL = 'Prognoza'
CLOSE_PRICE_LABEL = 'Cena zamknięcia'
PRICE_CHANGE_PCT_LABEL = 'Zmiana ceny (%)'
FORECAST_COUNT_LABEL = 'Liczba prognoz'
VALUE_CHANGE_LABEL = 'Zmiana wartości'

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
ROC_TITLE = 'ROC'

CLASS_ROC_LABEL = "Klasa '{0}' (obszar {1:0.2f})"
MICRO_AVG_ROC_LABEL = 'Mikro-średnia klas (obszar {0:0.2f})'
BINARY_ROC_LABEL = 'Krzywa ROC (obszar {0:0.2f})'


def plot_result(y_test, y_test_prediction, bench_params: BenchmarkParams, history, fpr, tpr, roc_auc, main_title):
    if bench_params.classes_count == 2:
        class_labels = [FALL_LABEL, RISE_LABEL]
        xticks = [0, 1]
        y_test_prediction[y_test_prediction >= 0.5] = 1
        y_test_prediction[y_test_prediction < 0.5] = 0
        y_test_prediction = to_categorical(y_test_prediction)
        y_test = to_categorical(y_test)
    else:
        class_labels = [FALL_LABEL, IDLE_LABEL, RISE_LABEL]
        xticks = [0, 1, 2]

    if len(y_test.shape) > 1:
        y_test = [np.argmax(pred, axis=None, out=None) for pred in y_test]
    y_test_prediction = [np.argmax(pred, axis=None, out=None) for pred in y_test_prediction]

    style.use('ggplot')

    plot_summary(y_test, y_test_prediction, bench_params, history, fpr, tpr, roc_auc, main_title, xticks, class_labels)

    if bench_params.plot_partial:
        plot_roc(bench_params, class_labels, fpr, roc_auc, tpr)
        plot_partial_and_close(bench_params, 'roc')

        plot_prediction_histogram(class_labels, xticks, y_test, y_test_prediction)
        plot_partial_and_close(bench_params, 'hist')

        if history is not None:
            plot_accuracy(bench_params, history)
            plot_partial_and_close(bench_params, 'acc')

            plot_loss(history)
            plot_partial_and_close(bench_params, 'loss')


def plot_partial_and_close(bench_params, prefix):
    plt.savefig(
        '{}/{}.pdf'.format(bench_params.save_partial_img_path,
                           benchmark_file_helper.get_patrial_img_path(bench_params, prefix)),
        format='pdf', dpi=1000)
    plt.savefig(
        '{}/{}.png'.format(bench_params.save_partial_img_path,
                           benchmark_file_helper.get_patrial_img_path(bench_params, prefix)))
    plt.close()


def plot_summary(y_test, y_test_prediction, bench_params: BenchmarkParams, history, fpr, tpr, roc_auc, main_title,
                 xticks, class_labels):
    if history is not None:
        plt.figure(figsize=(12, 12))
    else:
        plt.figure(figsize=(12, 6))
    plt.suptitle(main_title)
    if history is not None:
        plt.subplot(2, 2, 1)
    else:
        plt.subplot(1, 2, 1)
    plot_prediction_histogram(class_labels, xticks, y_test, y_test_prediction)
    if history is not None:
        plt.subplot(2, 2, 2)
    else:
        plt.subplot(1, 2, 2)
    plot_roc(bench_params, class_labels, fpr, roc_auc, tpr)
    if history is not None:
        plt.subplot(2, 2, 3)
        plot_loss(history)

        plt.subplot(2, 2, 4)
        plot_accuracy(bench_params, history)
    # plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('{}/{}.pdf'.format(bench_params.save_img_path, benchmark_file_helper.get_img_path(bench_params)),
                format='pdf', dpi=1000)
    plt.savefig('{}/{}.png'.format(bench_params.save_img_path, benchmark_file_helper.get_img_path(bench_params)))

    plt.close()


def plot_accuracy(bench_params, history):
    plt.plot(history.history[bench_params.metric])
    plt.plot(history.history['val_' + bench_params.metric])
    plt.title(ACCURACY_TITLE)
    plt.ylabel(ACCURACY_LABEL)
    plt.xlabel(EPOCH_LABEL)
    plt.legend([TRAIN_DATA, TEST_DATA], loc='upper left')


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(LOSS_TITLE)
    plt.ylabel(LOSS_LABEL)
    plt.xlabel(EPOCH_LABEL)
    plt.legend([TRAIN_DATA, TEST_DATA], loc='upper left')


def plot_roc(bench_params, class_labels, fpr, roc_auc, tpr):
    if bench_params.classes_count > 2:
        plt.plot(fpr[(MICRO_ROC_KEY)], tpr[MICRO_ROC_KEY],
                 label=MICRO_AVG_ROC_LABEL.format(roc_auc[MICRO_ROC_KEY]),
                 color='red', linewidth=3)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(bench_params.classes_count), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=CLASS_ROC_LABEL.format(class_labels[i], roc_auc[i]))
    else:
        plt.plot(fpr, tpr, label=BINARY_ROC_LABEL.format(roc_auc), color='red', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(FPR_LABEL)
    plt.ylabel(TPR_LABEL)
    plt.title(ROC_TITLE)
    plt.legend(loc="lower right")


def plot_prediction_histogram(class_labels, xticks, y_test, y_test_prediction):
    prediction_histogram = pd.DataFrame({'tmpcol': y_test})
    prediction_histogram['tmpcol'].plot(kind='hist', xticks=xticks, alpha=0.7, label=RATE_CHANGE_LABEL)
    prediction_histogram = pd.DataFrame.from_records({'tmpcol': y_test_prediction})
    prediction_histogram['tmpcol'].plot(kind='hist', xticks=xticks, alpha=0.7, label=RATE_CHANGE_LABEL)
    plt.legend([REAL_LEGEND, PREDICTED_LEGEND], loc='upper left')
    plt.xticks(xticks, class_labels)
    plt.xlabel(VALUE_CHANGE_LABEL)
    plt.ylabel(FORECAST_COUNT_LABEL)
    plt.title(HISTOGRAM_TITLE)
