# train/test split
# local score 3.6
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

    categoric_cols = [col for col in train.columns if train[col].dtype == 'object']

    if unique_id in categoric_cols:
        categoric_cols.remove(unique_id)

    # drop if too many values - usually a unique id column

    max_categories = train.shape[0] * 0.5

    too_many_value_categoric_cols = [col for col in categoric_cols
                                     if train[col].nunique() >= max_categories]

    if too_many_value_categoric_cols:
        print('dropping as too many categoric values', too_many_value_categoric_cols)

    categoric_cols = [i for i in categoric_cols if i not in too_many_value_categoric_cols]

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
            lbl.fit(list(train[col].values) + list(test[col].values))
            train[col] = lbl.transform(list(train[col].values))
            test[col] = lbl.transform(list(test[col].values))

    timer()

    return train, test

# Based on https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65


def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2

    NAlist = []  # Keeps track of columns that have missing values filled in.

    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings

            previous_type = df[col].dtype # pylint: disable=unused-variable

            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all():
                NAlist.append(col)
                df[col].fillna(mn - 1, inplace=True)

            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)

            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)

            # print(f"{col} before: {previous_type}, after: {df[col].dtype}")

    mem_usg = df.memory_usage().sum() / 1024**2

    print(f"Memory usage is: {mem_usg}MB, {100 * mem_usg / start_mem_usg:.1f}% of the initial size")

    return df, NAlist


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

    train, _ = reduce_mem_usage(train)
    test, _ = reduce_mem_usage(test)

    # obtain target and predictors

    target = 'meter_reading'
    unique_id = 'building_id'

    # drop timestamp for now
    train = train.drop('timestamp', axis=1)
    test = test.drop('timestamp', axis=1)

    # # numeric features
    # features = ['meter', 'square_feet']

    # train = train[features + [target, unique_id]].copy()
    # test = test[features + [unique_id]].copy()

    # train, test = clear_missing_values(train, test, ['year_built'], 0)

    # train, test = convert_numeric_categories(train, test, ['site_id'])

    # train, test = replace_missing_values(train, test, unique_id, target)

    train, test = encode_categoric_data(train, test, unique_id, target)

    test_predictions, train_score = evaluate(train, test, unique_id, target)

    print('score', train_score)

    # save predictions in format used for competition scoring
    output = pd.DataFrame({'row_id': test.index,
                           target: test_predictions})

    output.to_csv('submission.csv', index=False)

# -------- main


run()

print(f'Finished {((time() - start_time) / 60):.1f} mins\a')
