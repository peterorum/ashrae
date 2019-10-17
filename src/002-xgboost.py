# train/test split
# local score 3.6
# kaggle score
# minimize score

import pandas as pd
import numpy as np
import os
import sys
from pprint import pprint  # noqa
import warnings


from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

from time import time

is_kaggle = os.environ['HOME'] == '/tmp'
zipext = '' if is_kaggle else '.zip'

np.set_printoptions(threshold=sys.maxsize, precision=3, suppress=True)
pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore')

start_time = time()

# Read the data
# train = pd.read_csv(f'../input/ashrae-energy-prediction/sample.csv')
train = pd.read_csv(f'../input/ashrae-energy-prediction/train.csv{zipext}')
test = pd.read_csv(f'../input/ashrae-energy-prediction/test.csv{zipext}')

building_meta_data = pd.read_csv(f'../input/ashrae-energy-prediction/building_metadata.csv')

# match join on building_id

train = train.merge(building_meta_data, left_on = "building_id", right_on = "building_id", how = "left")
test = test.merge(building_meta_data, left_on = "building_id", right_on = "building_id", how = "left")

# Obtain target and predictors

target = 'meter_reading'

y = train[target]

numeric_features=[ 'meter', 'site_id', 'square_feet']

X = train[numeric_features].copy()
X_test = test[numeric_features].copy()

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

my_model = XGBRegressor(random_state=0)

# Fit the model to the training data & validate for score
my_model.fit(X_train, y_train)
preds_valid = my_model.predict(X_valid)
preds_valid = [ max(0, x) for x in preds_valid]

score = np.sqrt(mean_squared_error(np.log1p(y_valid), np.log1p(preds_valid)))

print(f'score: {score}')

# Generate test predictions on full set
my_model.fit(X, y)

preds_test = my_model.predict(X_test)

# Save predictions in format used for competition scoring
output = pd.DataFrame({'row_id': X_test.index,
                       target: preds_test})

output.to_csv('submission.csv', index=False)

print('%.0f mins' % ((time() - start_time) / 60))
