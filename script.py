import csv
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

dataset = np.zeros([165632, 3])
labels = []

csv_file = open("dataset-har-PUC-Rio-ugulino.csv")
csv_reader = csv.DictReader(csv_file, delimiter=';')

i=0

for row in csv_reader:
    try:
        x = int(row['x4'])
        y = int(row['y4'])
        z = int(row['z4'])
        dataset[i][0] = x
        dataset[i][1] = y
        dataset[i][2] = z
        labels.append(row['class'])
        """
        if row['class'] == 'sitting':
            labels.append(int(0))
        elif row['class'] == 'standing':
            labels.append(int(1))
        elif row['class'] == 'standingup':
            labels.append(int(2))
        elif row['class'] == 'sittingdown':
            labels.append(int(3))
        elif row['class'] == 'walking':
            labels.append(int(4))
        """
        i += 1
    except ValueError:
        continue

dataset = tf.keras.utils.normalize(dataset)
print dataset, 'length = ', len(dataset), '\n\n'

training_vals = dataset[::2]
testing_vals = dataset[1::2]


training_labels = np.asarray(labels)
testing_labels = np.asarray(labels)

training_labels = training_labels[0::2]
testing_labels = testing_labels[1::2]

le = preprocessing.LabelEncoder()

labels_encoded = le.fit_transform(labels)

model = KNeighborsClassifier()

model.fit(training_vals[0, :], training_vals[1, :])

"""
training_labels = np.asarray(labels)
testing_labels = np.asarray(labels)

training_labels = training_labels[0::2]
testing_labels = testing_labels[1::2]

print training_vals, 'length = ', len(training_vals), '\n\n'
print testing_vals, 'length = ', len(testing_vals), '\n\n'
print training_labels, 'length = ', len(training_labels), '\n\n'
print testing_labels, 'length = ', len(testing_labels), '\n\n'

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation=tf.nn.relu6),
    tf.keras.layers.Dense(64, activation=tf.nn.relu6),
    tf.keras.layers.Dense(32, activation=tf.nn.relu6),
    tf.keras.layers.Dense(16, activation=tf.nn.relu6),
    tf.keras.layers.Dense(5, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

model.fit(training_vals, training_labels, epochs=5)

test_loss, test_acc = model.evaluate(testing_vals, testing_labels)

print('Test accuracy:', test_acc)
"""

