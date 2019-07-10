import pandas as pd

CSV_TICKER = 'ticker'
CSV_DATE_COL = 'date'
CSV_TODAY_OPEN_COL = 'today_open'
CSV_TODAY_CLOSE_COL = 'today_close'
CSV_PREDICTION_COL = 'tommorow_prediction'
RESULT_PATH = './../../target/results/'

BUDGET = 100000
FEE = 0.005


def analyze_manual(filepath):
    cur_money = BUDGET
    buy_and_hold_money = BUDGET
    cur_securities = 0
    buy_and_hold_securities = 0

    df = pd.read_csv(RESULT_PATH + filepath)
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index)

    today_action = 1

    for day_ix in range(0, len(df)):

        day_date = pd.to_datetime(df.index.values[day_ix])
        today = df.iloc[day_ix]
        prediction = today[CSV_PREDICTION_COL]

        today_buy_price = today[CSV_TODAY_OPEN_COL] * (1 + FEE)
        today_sell_price = today[CSV_TODAY_OPEN_COL] * (1 - FEE)

        if day_ix == 0:
            buy_and_hold_securities = int(buy_and_hold_money / today_buy_price)
            buy_and_hold_money = buy_and_hold_money - buy_and_hold_securities * today_buy_price

        if today_action == 0 and cur_securities != 0:
            cur_money = cur_money + cur_securities * today_sell_price
            cur_securities = 0
        elif today_action == 2 and cur_securities == 0:
            cur_securities = int(cur_money / today_buy_price)
            cur_money = cur_money - cur_securities * today_buy_price

        if prediction == 0:
            prediction_str = 'sell'
        elif prediction == 2:
            prediction_str = 'buy'
        else:
            prediction_str = 'hold'

        print(
            '{0} wallet {1} securities and {2} dollars action for tommorow {3}. Buy and hold worth {4}.'.format(
                day_date.date(), cur_securities,
                round(cur_money, 2),
                prediction_str, round(buy_and_hold_money + buy_and_hold_securities*today_sell_price,2)))

        today_action = prediction

    print('Finished ' + filepath)


if __name__ == '__main__':
    analyze_manual('GOOGL.csv')
    analyze_manual('INTC.csv')
    analyze_manual('MSFT.csv')
    print('Finished all')
