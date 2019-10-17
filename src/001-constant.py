# baseline: constant 0
# local score 4.668
# kaggle score 4.69

import sys  # pylint: disable=unused-import
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from time import time

import os

is_kaggle = os.environ['HOME'] == '/tmp'

zipext = '' if is_kaggle else '.zip'

# load data
train = pd.read_csv(f'../input/train.csv{zipext}')
test = pd.read_csv(f'../input/test.csv{zipext}')

#-------- main

start_time = time()

target = 'meter_reading'

result = 0

train['predicted'] = result

score = np.sqrt(mean_squared_error(np.log1p(train[target]), np.log1p(train.predicted)))

print('score', score)

test[target] = result

predictions = test[['row_id', target]]

predictions.to_csv('submission.csv', index=False)

print('%.0f mins' % ((time() - start_time) / 60))
