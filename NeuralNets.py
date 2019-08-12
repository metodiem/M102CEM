from __future__ import absolute_import, division, print_function
import certifi
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import csv
import os
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

certifi.where()
tf.enable_eager_execution()

#print("TensorFlow version: {}".format(tf.__version__))

train_dataset_url1 = "D:\\University\\Master\\M102CEM - Project\\Code\\Converted.csv"

#dataset = np.loadtxt(path, delimiter=',', dtype=np.int32) #numpy loading of a csv file

#train_input_fn = tf.estimator.inputs.numpy_input_fn(
#    x={"x": np.array(training_set.data)},
#    y=np.array(training_set.target),
#    num_epochs=None,
#    shuffle=True)

train_dataset_url = "https://www.dropbox.com/s/4s16ehrnfohjz5s/TestData.csv"
#train_dataset_url = "https://iotanalytics.unsw.edu.au/iottestbed/csv/16-09-23.csv.zip"
#train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
column_names = ['time', 'size', 'eth.src', 'eth.dst', 'ip.src', 'ip.dst', 'ip.proto', 'port.src', 'port.dst', 'device']
#column_names = ['time', 'size','device']
print("pesssssss")
#dataset = tf.keras.utils.get_dataset(train_dataset_url, column_names)
#dataset = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url), origin=train_dataset_url, extract=True)
#record_defaults = [tf.int32] * 3   # Eight required float columns
#dataset = tf.data.experimental.CsvDataset(path, record_defaults)

#print(dataset)



features_names = column_names[:-1]
label_name = column_names[-1]

print("Features: {}".format(features_names))
print("Label: {}".format(label_name))

batch_size = 32
#print(dataset)
#print(type(dataset))

train_dataset = tf.contrib.data.make_csv_dataset(
    train_dataset_url1,
    batch_size,
    column_names = column_names,
    label_name = label_name,
    num_epochs=1)



features, labels = next(iter(train_dataset))

print(features)

#plt.scatter(features['ip.src'].numpy(),
#            features['size'].numpy(),
#            c=labels.numpy(),
#            cmap='viridis')
#
#plt.xlabel("ip.src")
#plt.ylabel("Size")
#plt.show()

def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

train_dataset = train_dataset.map(pack_features_vector)
features, labels = next(iter(train_dataset))

print(features[:5])

model = tf.keras.Sequential([
  tf.keras.layers.Dense(100, activation=tf.nn.relu, input_shape=(9,)),  # input shape required
  tf.keras.layers.Dense(100, activation=tf.nn.relu),
  tf.keras.layers.Dense(31)
])

predictions = model(features)
predictions[:5]

tf.nn.softmax(predictions[:5])

print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
print("    Labels: {}".format(labels))