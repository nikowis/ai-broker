import os
from itertools import cycle

import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.metrics import roc_curve, auc

import db.stock_constants as const

BASE_IMG_PATH = './../../target'
if not os.path.exists(BASE_IMG_PATH):
    os.makedirs(BASE_IMG_PATH)

OUTSTANDING_PATH = BASE_IMG_PATH + '/outstanding'
if not os.path.exists(OUTSTANDING_PATH):
    os.makedirs(OUTSTANDING_PATH)

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
CLASS_LABELS = [FALL_LABEL, IDLE_LABEL, RISE_LABEL]
PREDICTED_LEGEND = 'Przewidywane'
REAL_LEGEND = 'Rzeczywiste'

REAL_VALUES_LABEL = 'Rzeczywiste wartości '
PREDICTED_VALUES_LABEL = 'Przewidywane wartości '
EPOCH_LABEL = 'iteracja'
ACCURACY_LABEL = 'dokładność (%)'
LOSS_LABEL = 'wartość funkcji straty'
LOSS_TITLE = 'Strata'
ACCURACY_TITLE = 'Dokładność'
TEST_DATA = 'Dane testowe'
TRAIN_DATA = 'Dane treningowe'

#ROC
FPR_LABEL = 'False Positive Rate'
TPR_LABEL = 'True Positive Rate'
ROC_TITLE = 'Krzywe ROC'
MICRO_ROC_KEY = "micro"
CLASS_ROC_LABEL = "Klasa '{0}' (obszar = {1:0.2f})"
MICRO_AVG_ROC_LABEL = 'Mikro-średnia klas (obszar = {0:0.2f})'

def legend_labels_save_files(title, file_name='img', base_img_path=BASE_IMG_PATH, xlabel=DATE_LABEL,
                             ylabel=CLOSE_PRICE_USD_LABEL, legend=4):
    if not legend == -1:
        plt.legend(loc=legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig('{}/{}.eps'.format(base_img_path, file_name), format='eps', dpi=1000)
    plt.savefig('{}/{}.png'.format(base_img_path, file_name))
    plt.show()
    plt.close()


def calculate_roc_auc(y_test, y_test_score, classes_count):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(classes_count):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_test_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr[MICRO_ROC_KEY], tpr[MICRO_ROC_KEY], _ = roc_curve(y_test.ravel(), y_test_score.ravel())
    roc_auc[MICRO_ROC_KEY] = auc(fpr[MICRO_ROC_KEY], tpr[MICRO_ROC_KEY])

    return fpr, tpr, roc_auc


def plot_result(df,y_test, y_test_score, classes_count, history, main_title, file_name, outstanding=False):
    fpr, tpr, roc_auc = calculate_roc_auc(y_test,y_test_score, classes_count)

    plt.figure(figsize=(12, 12))
    style.use('ggplot')
    plt.suptitle(main_title)

    plt.subplot(2, 2, 1)
    df[const.LABEL_DISCRETE_COL].plot(kind='hist', xticks=[0, 1, 2],alpha=0.7, label=RATE_CHANGE_LABEL)
    df[const.FORECAST_DISCRETE_COL].plot(kind='hist', xticks=[0, 1, 2],alpha=0.7, label=RATE_CHANGE_FORECAST_LABEL)
    plt.legend([REAL_LEGEND, PREDICTED_LEGEND], loc='upper left')
    plt.xticks([0, 1, 2], CLASS_LABELS)
    plt.xlabel(VALUE_CHANGE_LABEL)
    plt.ylabel(FORECAST_COUNT_LABEL)
    plt.title(REAL_VALUES_LABEL)

    plt.subplot(2, 2, 2)
    plt.plot(fpr[(MICRO_ROC_KEY)], tpr[MICRO_ROC_KEY],
             label=MICRO_AVG_ROC_LABEL.format(roc_auc[MICRO_ROC_KEY]),
             color='red', linewidth=3)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(classes_count), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=CLASS_ROC_LABEL.format(CLASS_LABELS[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(FPR_LABEL)
    plt.ylabel(TPR_LABEL)
    plt.title(ROC_TITLE)
    plt.legend(loc="lower right")

    plt.subplot(2, 2, 3)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(LOSS_TITLE)
    plt.ylabel(LOSS_LABEL)
    plt.xlabel(EPOCH_LABEL)
    plt.legend([TRAIN_DATA, TEST_DATA], loc='upper left')

    plt.subplot(2, 2, 4)
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title(ACCURACY_TITLE)
    plt.ylabel(ACCURACY_LABEL)
    plt.xlabel(EPOCH_LABEL)
    plt.legend([TRAIN_DATA, TEST_DATA], loc='upper left')

    # plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if outstanding:
        # plt.savefig('{}/{}.eps'.format(OUTSTANDING_PATH, file_name), format='eps', dpi=1000)
        plt.savefig('{}/{}.png'.format(OUTSTANDING_PATH, file_name))
    else:
        # plt.savefig('{}/{}.eps'.format(BASE_IMG_PATH, file_name), format='eps', dpi=1000)
        plt.savefig('{}/{}.png'.format(BASE_IMG_PATH, file_name))
    plt.show()
    plt.close()
