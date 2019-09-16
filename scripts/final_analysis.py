import os
import pandas as pd

DIR = './../predictions/'

if __name__ == "__main__":

    df_list = []
    sym_list = []
    for filename in os.listdir(DIR):
        if filename.endswith(".csv"):
            print(filename)
            df = pd.read_csv(DIR + filename)
            print(df.head())

    print("Program finished")
