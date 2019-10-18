# train/test split
# local score 3.6
# kaggle score 3.06
# minimize score

import os
import sys
from pprint import pprint  # noqa
import warnings
from time import time

import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


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

    model = XGBRegressor(objective='reg:squarederror', random_state=0)

    train_inputs = train.drop([unique_id, target], axis=1)

    x_train, x_validate, y_train, y_validate = train_test_split(
        train_inputs, train[target], test_size=0.2, random_state=1)

    model.fit(x_train, y_train)

    train_predictions = model.predict(x_validate)

    train_predictions = [max(0, x) for x in train_predictions]

    train_score = np.sqrt(mean_squared_error(
        np.log1p(train_predictions), np.log1p(y_validate)))

    test_predictions = model.predict(test[x_train.columns])

    test_predictions = [max(0, x) for x in test_predictions]

    timer()

    return test_predictions, train_score


# clear empty values that should not get a mean
# may be run twice for numeric values (0) then categorical ('NA')


def clear_missing_values(train, test, columns, value):

    for col in columns:
        train[col] = train[col].fillna(value)
        test[col] = test[col].fillna(value)

    return train, test


# convert numeric columns which are actually just categories to string


def convert_numeric_categories(train, test, columns):

    for col in columns:
        train[col] = train[col].apply(str)
        test[col] = test[col].apply(str)

    return train, test


# --------------------- run


def run():

    # read the data

    if is_kaggle:
        train = pd.read_csv(
            f'../input/ashrae-energy-prediction/train.csv{zipext}')
        test = pd.read_csv(
            f'../input/ashrae-energy-prediction/test.csv{zipext}')
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

    # obtain target and predictors

    target = 'meter_reading'
    unique_id = 'building_id'

    # numeric features
    features = ['meter', 'square_feet']

    train = train[features + [target, unique_id]].copy()
    test = test[features + [unique_id]].copy()

    # train, test = clear_missing_values(train, test, ['year_built'], 0)

    # train, test = convert_numeric_categories(train, test, ['site_id'])

    test_predictions, train_score = evaluate(train, test, unique_id, target)

    print('score', train_score)

    # save predictions in format used for competition scoring
    output = pd.DataFrame({'row_id': test.index,
                           target: test_predictions})

    output.to_csv('submission.csv', index=False)

# -------- main


run()

print(f'Finished {((time() - start_time) / 60):.1f} mins\a')
