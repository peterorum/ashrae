# create sample of data

import pandas as pd

test = pd.read_csv("../input/ashrae-energy-prediction/test.csv.zip")

sample_size = 10000

sample = test.sample(sample_size)

sample.to_csv('../input/test-sample.csv', index=False)
