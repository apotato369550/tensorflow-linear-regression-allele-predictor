from __future__ import absolute_import, division, print_function, unicode_literals
from msilib.schema import FeatureComponents

import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

data_frame_train = pd.read_csv("train.csv")
data_frame_eval = pd.read_csv("eval.csv")

print(data_frame_eval.iloc[0]["p1_alleles"])

print(data_frame_eval.columns.tolist())
print(data_frame_train.columns.tolist())

print(data_frame_train.dtypes)
print(data_frame_eval.dtypes)


y_train = data_frame_train.pop('child_dominant')
y_eval = data_frame_eval.pop('child_dominant')

CATEGORICAL_COLUMNS = ["p1_alleles", "p2_alleles", "child_alleles"]
NUMERIC_COLUMNS = ["p1_dominant", "p2_dominant"]

feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = data_frame_train[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
  
for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

def make_input_function(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_function = make_input_function(data_frame_train, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_function = make_input_function(data_frame_eval, y_eval, num_epochs=1, shuffle=False)

linear_estimate_model = tf.estimator.LinearClassifier(feature_columns=feature_columns)



linear_estimate_model.train(train_input_function) 

result = linear_estimate_model.evaluate(eval_input_function)


print("Model Accuracy: ")

result = list(linear_estimate_model.predict(eval_input_function))

print(result[0])

clear_output()

# print("Prediction Accuracy: " + str(result["accuracy"] * 100))

for i in range(5):
  print("Child #" + str(i + 1) + "\n")
  print("Parents:")
  print("Parent 1 - Alleles: " + str(data_frame_eval.iloc[i]["p1_alleles"]) + " Dominant Hand: " + str("Right" if data_frame_eval.iloc[i]["p1_dominant"] else "Left"))
  print("Parent 2 - Alleles: " + str(data_frame_eval.iloc[i]["p2_alleles"]) + " Dominant Hand: " + str("Right" if data_frame_eval.iloc[i]["p2_dominant"] else "Left"))
  print("\nChild Alleles: " + str(data_frame_eval.iloc[i]["child_alleles"]))
  print("\nProbability of being right handed: %" + str(round(100 * result[i]['probabilities'][1], 2)))
  print("Probability of being left handed: %" + str(round(100 * result[i]['probabilities'][0], 2)) + "\n")

