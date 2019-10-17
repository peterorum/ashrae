# train/test split
# local score 3.6
# kaggle score 3.06
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
last_time = time()


def timer():
    global last_time

    print(f'{((time() - last_time) / 60):.1f}, {((time() - start_time) / 60):.1f} mins\n')  # noqa

    last_time = time()

# run the model


def evaluate(train, test, unique_id, target):

    print('evaluate')

    model = XGBRegressor(objective ='reg:squarederror', random_state=0)

    train_inputs = train.drop([unique_id, target], axis=1)

    x_train, x_validate, y_train, y_validate = train_test_split(
        train_inputs, train[target], test_size=0.2, random_state=1)

    model.fit(x_train, y_train)

    train_predictions = model.predict(x_validate)

    train_predictions = [ max(0, x) for x in train_predictions]

    train_score = np.sqrt(mean_squared_error(np.log1p(train_predictions), np.log1p(y_validate)))

    test_predictions = model.predict(test[x_train.columns])

    test_predictions = [ max(0, x) for x in test_predictions]

    timer()

    return test_predictions, train_score


# --------------------- run


def run():

    # read the data

    if is_kaggle:
        train = pd.read_csv(
            '../input/ashrae-energy-prediction/train.csv{zipext}')
        test = pd.read_csv(
            '../input/ashrae-energy-prediction/test.csv{zipext}')
    else:
        train = pd.read_csv(
            '../input/ashrae-energy-prediction/sample-train.csv')
        test = pd.read_csv('../input/ashrae-energy-prediction/sample-test.csv')

    building_meta_data = pd.read_csv(
        '../input/ashrae-energy-prediction/building_metadata.csv')

    # match join on building_id

    train = train.merge(building_meta_data, left_on="building_id",
                        right_on="building_id", how="left")

    test = test.merge(building_meta_data, left_on="building_id",
                      right_on="building_id", how="left")

    original_columns = train.columns.tolist()

    # obtain target and predictors

    target = 'meter_reading'
    unique_id = 'building_id'

    y = train[target]

    # numeric features
    features = ['meter', 'site_id', 'square_feet']

    X = train[features + [target, unique_id]].copy()
    X_test = test[features + [unique_id]].copy()

    test_predictions, train_score = evaluate(X, X_test, unique_id, target)

    print('score', train_score)

    # save predictions in format used for competition scoring
    output = pd.DataFrame({'row_id': X_test.index,
                           target: test_predictions})

    output.to_csv('submission.csv', index=False)

# -------- main


run()

print(f'Finished {((time() - start_time) / 60):.1f} mins\a')
