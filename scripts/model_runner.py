import os
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import data_preprocessing
import db_access
import nn_model
import plot_helper
import stock_constants as const
from sklearn import model_selection

MIN_DATE = '2009-01-01'
MAX_DATE = '2020-10-29'
SELECTED_SYM = 'GOOGL'

def run(model,x_train, x_test, y_train, y_test, epochs=10, batch_size=5):
    iter_time = time.time()
    iteration=0

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        epochs=epochs,
                        batch_size=batch_size, verbose=0)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("Loss: ", loss, " Accuracy: ", accuracy, " epochs: ", epochs)

    print('Time ', str(int(time.time() - iter_time)), 's.')

    y_test_score = model.predict(x_test)



if __name__ == '__main__':
    db_conn = db_access.create_db_connection(remote=False)
    df_list, sym_list = db_access.find_by_tickers_to_dateframe_parse_to_df_list(db_conn, [SELECTED_SYM],
                                                                                min_date=MIN_DATE, max_date=MAX_DATE)

    df = df_list[0]

    df,x,y = data_preprocessing.preprocess(df)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, shuffle=False)

    model = nn_model.create_seq_model([5, 5], input_size=x_train.shape[1], activation='relu',
                                      optimizer='adam',
                                      loss='binary_crossentropy', class_count=1)
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, encoded_Y, test_size=0.2, shuffle=False)

    run(model, x_train, x_test, y_train, y_test)

    # https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
    # seed = 7
    # np.random.seed(seed)
    # estimator = KerasClassifier(build_fn=model, epochs=10, batch_size=5, verbose=0)
    # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    # results = cross_val_score(estimator, x, encoded_Y, cv=kfold)
    # print("Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
