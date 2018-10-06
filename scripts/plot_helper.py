import matplotlib.pyplot as plt

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


def legend_labels_save_files(title, file_name='img', base_img_path='/target', xlabel=DATE_LABEL,
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
