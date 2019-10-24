# train/test split
# local score 2.35
# kaggle score 2.36
# minimize score

import os
import sys
from pprint import pprint  # pylint: disable=unused-import
import warnings
from time import time

import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
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


# --- categoric data


def encode_categoric_data(train, test, unique_id, target):

    print('encode_categoric_data')

    train_targets = train[target]

    categoric_cols = [
        col for col in train.columns if train[col].dtype == 'object']

    if unique_id in categoric_cols:
        categoric_cols.remove(unique_id)

    # drop if too many values - usually a unique id column

    max_categories = train.shape[0] * 0.5

    too_many_value_categoric_cols = [col for col in categoric_cols
                                     if train[col].nunique() >= max_categories]

    if too_many_value_categoric_cols:
        print('dropping as too many categoric values',
              too_many_value_categoric_cols)

    categoric_cols = [
        i for i in categoric_cols if i not in too_many_value_categoric_cols]

    train = train.drop(too_many_value_categoric_cols, axis=1)
    test.drop([col for col in too_many_value_categoric_cols
               if col in test.columns], axis=1, inplace=True)

    # one-hot encode if not too many values

    max_ohe_categories = 15

    ohe_categoric_cols = [col for col in categoric_cols
                          if train[col].nunique() <= max_ohe_categories]

    categoric_cols = [i for i in categoric_cols if i not in ohe_categoric_cols]

    if ohe_categoric_cols:
        print('one-hot encode', ohe_categoric_cols)

        # one-hot encode & align to have same columns
        train = pd.get_dummies(train, columns=ohe_categoric_cols)
        test = pd.get_dummies(test, columns=ohe_categoric_cols)
        train, test = train.align(test, join='inner', axis=1)

        # restore after align
        train[target] = train_targets

    # possibly rank encode rather than ohe. see gstore.

    # label encode the remainder (convert to integer)

    label_encode_categoric_cols = categoric_cols

    if label_encode_categoric_cols:
        print('label encode', label_encode_categoric_cols)

        for col in label_encode_categoric_cols:
            lbl = LabelEncoder()
            # lbl.fit(list(train[col].values) + list(test[col].values))
            # train[col] = lbl.transform(list(train[col].values))
            # test[col] = lbl.transform(list(test[col].values))

            # lbl.fit(list(train[col].astype(str)) + list(test[col].astype(str)))
            # train[col] = lbl.transform(list(train[col].astype(str))).astype(np.int8)
            # test[col] = lbl.transform(list(test[col].astype(str))).astype(np.int8)

            train[col] = lbl.fit_transform(train[col].astype(str)).astype(np.int8)
            test[col] = lbl.fit_transform(test[col].astype(str)).astype(np.int8)

    timer()

    return train, test


def reduce_mem_usage(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    print('mem usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
        end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

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

    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)

    # obtain target and predictors

    target = 'meter_reading'
    unique_id = 'building_id'

    # drop timestamp for now
    train = train.drop('timestamp', axis=1)
    test = test.drop('timestamp', axis=1)

    # # numeric features
    features = ['meter', 'square_feet']

    train = train[features + [target, unique_id]].copy()
    test = test[features + [unique_id]].copy()

    # train, test = clear_missing_values(train, test, ['year_built'], 0)

    # train, test = convert_numeric_categories(train, test, ['site_id'])

    # train, test = replace_missing_values(train, test, unique_id, target)

    # train, test = encode_categoric_data(train, test, unique_id, target)

    test_predictions, train_score = evaluate(train, test, unique_id, target)

    print('score', train_score)

    # save predictions in format used for competition scoring
    output = pd.DataFrame({'row_id': test.index,
                           target: test_predictions})

    output.to_csv('submission.csv', index=False)

# -------- main


run()

print(f'Finished {((time() - start_time) / 60):.1f} mins\a')
