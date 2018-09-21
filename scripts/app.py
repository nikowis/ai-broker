import math

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import alpha as stck
import model_runner as model_runner

alpha = stck.AlphaVantage()
df = alpha.daily_adjusted('GOOGL')
df = alpha.daily_adjusted('CSCO')

PCT_CHANGE = 'PCT_change'
HL_PCT = 'HL_PCT'

df = df[[alpha.OPEN_COL, alpha.HIGH_COL, alpha.LOW_COL, alpha.ADJUSTED_CLOSE_COL, alpha.VOLUME_COL]]

df[HL_PCT] = (df[alpha.HIGH_COL] - df[alpha.LOW_COL]) / df[alpha.LOW_COL] * 100.0
df[PCT_CHANGE] = (df[alpha.ADJUSTED_CLOSE_COL] - df[alpha.OPEN_COL]) / df[alpha.OPEN_COL] * 100.0
df = df[[alpha.ADJUSTED_CLOSE_COL, HL_PCT, PCT_CHANGE, alpha.VOLUME_COL]]
one_percent = int(math.ceil(0.01 * len(df)))
predictor = LinearRegression(n_jobs=-1)
df = model_runner.predict_exact(predictor, df, 1)

df[alpha.ADJUSTED_CLOSE_COL].plot(label='Cena zamknięcia')
#df['Forecast'].plot(label='Prognoza')
plt.legend(loc=4)
plt.xlabel('Data')
plt.ylabel('Cena zamknięcia (USD)')
plt.savefig('target/plot.png')
plt.show()
