# import csv
# import numpy as np
# import tensorflow as tf
# from sklearn import preprocessing
# from sklearn.neighbors import KNeighborsClassifier
#
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#
# dataset = np.zeros([165632, 3])
# labels = []
#
# csv_file = open("dataset-har-PUC-Rio-ugulino.csv")
# csv_reader = csv.DictReader(csv_file, delimiter=';')
#
# i=0
#
# for row in csv_reader:
#     try:
#         x = int(row['x4'])
#         y = int(row['y4'])
#         z = int(row['z4'])
#         dataset[i][0] = x
#         dataset[i][1] = y
#         dataset[i][2] = z
#         labels.append(row['class'])
#         """
#         if row['class'] == 'sitting':
#             labels.append(int(0))
#         elif row['class'] == 'standing':
#             labels.append(int(1))
#         elif row['class'] == 'standingup':
#             labels.append(int(2))
#         elif row['class'] == 'sittingdown':
#             labels.append(int(3))
#         elif row['class'] == 'walking':
#             labels.append(int(4))
#         """
#         i += 1
#     except ValueError:
#         continue
#
# dataset = tf.keras.utils.normalize(dataset)
# print dataset, 'length = ', len(dataset), '\n\n'
#
# training_vals = dataset[::2]
# testing_vals = dataset[1::2]
#
#
# training_labels = np.asarray(labels)
# testing_labels = np.asarray(labels)
#
# training_labels = training_labels[0::2]
# testing_labels = testing_labels[1::2]
#
# le = preprocessing.LabelEncoder()
#
# labels_encoded = le.fit_transform(labels)
#
# model = KNeighborsClassifier()
#
# model.fit(training_vals[0, :], training_vals[1, :])
#
# """
# training_labels = np.asarray(labels)
# testing_labels = np.asarray(labels)
#
# training_labels = training_labels[0::2]
# testing_labels = testing_labels[1::2]
#
# print training_vals, 'length = ', len(training_vals), '\n\n'
# print testing_vals, 'length = ', len(testing_vals), '\n\n'
# print training_labels, 'length = ', len(training_labels), '\n\n'
# print testing_labels, 'length = ', len(testing_labels), '\n\n'
#
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(128, activation=tf.nn.relu6),
#     tf.keras.layers.Dense(64, activation=tf.nn.relu6),
#     tf.keras.layers.Dense(32, activation=tf.nn.relu6),
#     tf.keras.layers.Dense(16, activation=tf.nn.relu6),
#     tf.keras.layers.Dense(5, activation=tf.nn.softmax)
# ])
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['sparse_categorical_accuracy'])
#
# model.fit(training_vals, training_labels, epochs=5)
#
# test_loss, test_acc = model.evaluate(testing_vals, testing_labels)
#
# print('Test accuracy:', test_acc)
# """

# cnn model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
	print(trainX.shape, trainy.shape)
	# load all test
	testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
	print(testX.shape, testy.shape)
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	# one hot encode y
	trainy = to_categorical(trainy)
	testy = to_categorical(testy)
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	return trainX, trainy, testX, testy

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
	verbose, epochs, batch_size = 0, 10, 32
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
	model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
	model.add(Dropout(0.5))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=10):
	# load data
	trainX, trainy, testX, testy = load_dataset()
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_model(trainX, trainy, testX, testy)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)

# run the experiment
run_experiment()