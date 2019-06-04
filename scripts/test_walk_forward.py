import math

import numpy as np

TEST_SIZE = 0.4

TEST_WINDOW_SIZE = 15
TRAIN_WINDOW_SIZE = 100
RETRAIN_EPOCHS = 10

print('Start')

df = np.arange(200)

x_trains_list = []
x_tests_list = []

for i in range(0, int((TEST_SIZE * len(df) / TEST_WINDOW_SIZE))):

    train_end_idx = int((1-TEST_SIZE) * len(df) + i * TEST_WINDOW_SIZE)
    train_start_idx = int(max(0, train_end_idx-TRAIN_WINDOW_SIZE))
    test_start_idx = int(train_end_idx + 1)
    test_end_idx = int(train_end_idx + TEST_WINDOW_SIZE)

    x_train=df[train_start_idx:train_end_idx+1]
    x_test=df[test_start_idx:test_end_idx+1]
    x_trains_list.append(x_train)
    x_tests_list.append(x_test)
    print('i: ', i, ' train start idx: ', train_start_idx, ' train end id:x ', train_end_idx, ' test start idx: ',
          test_start_idx, ' test end idx: ', test_end_idx)



print('Finished')
