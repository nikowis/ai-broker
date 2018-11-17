import os

import matplotlib.pyplot as plt
from matplotlib import style

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
REAL_VALUES_LABEL = 'Rzeczywiste wartości '
PREDICTED_VALUES_LABEL = 'Przewidywane wartości '
EPOCH_LABEL = 'iteracja'
ACCURACY_LABEL = 'dokładność'
LOSS_LABEL = 'wartość funkcji straty'
LOSS_TITLE = 'Strata modelu'
ACCURACY_TITLE = 'Dokładność modelu'


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


def plot_result(df, history, main_title, file_name, outstanding=False):
    fig = plt.figure(figsize=(12, 12))
    style.use('ggplot')

    plt.suptitle(main_title)
    plt.subplot(2, 2, 1)
    df[const.LABEL_DISCRETE_COL].plot(kind='hist', xticks=[0, 1, 2], label=RATE_CHANGE_LABEL)
    plt.xticks([0, 1, 2], [FALL_LABEL, IDLE_LABEL, RISE_LABEL])
    plt.xlabel(VALUE_CHANGE_LABEL)
    plt.ylabel(FORECAST_COUNT_LABEL)
    plt.title(REAL_VALUES_LABEL)
    plt.subplot(2, 2, 2)
    df[const.FORECAST_DISCRETE_COL].plot(kind='hist', xticks=[0, 1, 2], label=RATE_CHANGE_FORECAST_LABEL)
    plt.xticks([0, 1, 2], [FALL_LABEL, IDLE_LABEL, RISE_LABEL])
    plt.xlabel(VALUE_CHANGE_LABEL)
    plt.ylabel(FORECAST_COUNT_LABEL)
    plt.title(PREDICTED_VALUES_LABEL)
    plt.subplot(2, 2, 3)
    plt.plot(history.history['loss'])
    plt.title(LOSS_TITLE)
    plt.ylabel(LOSS_LABEL)
    plt.xlabel(EPOCH_LABEL)
    plt.subplot(2, 2, 4)
    plt.plot(history.history['categorical_accuracy'])
    plt.title(ACCURACY_TITLE)
    plt.ylabel(ACCURACY_LABEL)
    plt.xlabel(EPOCH_LABEL)
    # plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if outstanding:
        #plt.savefig('{}/{}.eps'.format(OUTSTANDING_PATH, file_name), format='eps', dpi=1000)
        plt.savefig('{}/{}.png'.format(OUTSTANDING_PATH, file_name))
    else:
        #plt.savefig('{}/{}.eps'.format(BASE_IMG_PATH, file_name), format='eps', dpi=1000)
        plt.savefig('{}/{}.png'.format(BASE_IMG_PATH, file_name))
    #plt.show()
    plt.close()
