from __future__ import absolute_import, division, print_function
import certifi
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Remove the warning
import csv
import os
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

certifi.where()
tf.enable_eager_execution()


train_dataset_url = "D:\\University\\Master\\M102CEM - Project\\Code\\Converted.csv"


#train_dataset_url = "https://www.dropbox.com/s/4s16ehrnfohjz5s/TestData.csv"
column_names = ['time','size', 'ip.proto', 'port.src', 'port.dst', 'device']
#dataset = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url), origin=train_dataset_url, extract=True)
#record_defaults = [tf.int32] * 3   # Eight required float columns




features_names = column_names[:-1]
label_name = column_names[-1]

print("Features: {}".format(features_names))
print("Label: {}".format(label_name))

batch_size = 32

#Create a tensor dataset
train_dataset = tf.contrib.data.make_csv_dataset( 
    train_dataset_url,
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

"""
Create a neural network with an input layer that consists of 5 neurons,
two hidden layers with 12 neurons each and
an output layer with 31 neurons which is the total number of devices we have on the network
"""
model = tf.keras.Sequential([
  tf.keras.layers.Dense(12, activation=tf.nn.relu, input_shape=(5,)),  
  tf.keras.layers.Dense(12, activation=tf.nn.relu),
  tf.keras.layers.Dense(31)
])

predictions = model(features)
predictions[:5]

tf.nn.softmax(predictions[:5])

print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
print("    Labels: {}".format(labels))

def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


l = loss(model, features, labels)
print("Loss test: {}".format(l))

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)

global_step = tf.Variable(0)

loss_value, grads = grad(model, features, labels)

print("Step: {}, Initial Loss: {}".format(global_step.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)

print("Step: {},         Loss: {}".format(global_step.numpy(),
                                          loss(model, features, labels).numpy()))

## Note: Rerunning this cell uses the same model variables

from tensorflow import contrib
tfe = contrib.eager

# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
  epoch_loss_avg = tfe.metrics.Mean()
  epoch_accuracy = tfe.metrics.Accuracy()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables),
                              global_step)

    # Track progress
    epoch_loss_avg(loss_value)  # add current batch loss
    # compare predicted label to actual label
    epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

  # end epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 10 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))
#Random traffic from the list
predict_dataset = tf.convert_to_tensor([
    [150004, 66, 6, 37802, 1935,],
])

predictions = model(predict_dataset)
#Make a prediction
for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx] 
  print(i, 100*p)
