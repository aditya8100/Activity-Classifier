import matplotlib.pyplot as plt
import keras
import numpy as np
import pandas as pd
#from IPython.display import display, HTML
from scipy import stats

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
# import os
#
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def read_data(filename):
    col_names = ['user',
                 'activity',
                 'timestamp',
                 'x',
                 'y',
                 'z']

    df = pd.read_csv(filename, header=None, names=col_names)

    df['z'].replace(regex=True,
                    inplace=True,
                    to_replace=r';',
                    value=r'')

    df['z'] = df['z'].apply(convert_to_float)
    df.dropna(axis=0, how='any', inplace=True)

    return df


def convert_to_float(num):
    try:
        return np.float(num)

    except:
        return np.nan


def segment_and_labels (dframe, time_step, step):
    N_FEATURES = 3

    segments = []
    labels = []

    for i in range(0, len(dframe) - time_step, step):
        xs = dframe['x'].values[i:i+time_step]
        ys = dframe['y'].values[i:i+time_step]
        zs = dframe['z'].values[i:i+time_step]

        label = stats.mode(dframe['ActivityEncoded'].values[i:i+time_step], axis=None)[0][0]
        segments.append([xs, ys, zs])
        labels.append(label)

    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_step, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels




pd.options.display.float_format = '{:.1f}'.format

plt.style.use('ggplot')

#print 'Keras version ', keras.__version__

LABELS = ['Downstairs',
          'Jogging',
          'Sitting',
          'Standing',
          'Upstairs',
          'Walking']

TIME_PERIODS = 80
STEP_VAL = 40

df = read_data('WISDM_ar_v1.1_raw.txt')

print 'Number of rows: ', df.shape[0]
print 'Number of cols: ', df.shape[1]

print df.head(20)

df['activity'].value_counts().plot(kind='bar')
# plt.show()

LABEL = 'ActivityEncoded'
le = preprocessing.LabelEncoder()

df[LABEL] = le.fit_transform(df['activity'].values.ravel())

df_test = df[df['user'] > 28]
df_train = df[df['user'] <= 28]

pd.options.mode.chained_assignment = None  # default='warn'

df_train['x'] = df_train['x'] / df_train['x'].max()
df_train['y'] = df_train['y'] / df_train['y'].max()
df_train['z'] = df_train['z'] / df_train['z'].max()

df_train = df_train.round({'x': 4, 'y': 4, 'z': 4})

segments, labels = segment_and_labels(df_train, TIME_PERIODS, STEP_VAL)

print segments.shape, labels.shape

num_time_periods, num_sensors = segments.shape[1], segments.shape[2]
num_classes = le.classes_.size

segments = segments.reshape(segments.shape[0], num_time_periods * num_sensors)

segments = segments.astype('float32')
labels = labels.astype('float32')

labels_hot = keras.utils.np_utils.to_categorical(labels, num_classes)

model = keras.Sequential()

model.add(keras.layers.Reshape((TIME_PERIODS, num_sensors), input_shape=(num_time_periods * num_sensors, )))
model.add(keras.layers.Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
model.add(keras.layers.Conv1D(100, 10, activation='relu'))
model.add(keras.layers.MaxPool1D(3))
model.add(keras.layers.Conv1D(160, 10, activation='relu'))
model.add(keras.layers.Conv1D(160, 10, activation='relu'))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(num_classes, activation='softmax'))

print model.summary()

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
]

model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

BATCH_SIZE = 400
EPOCHS = 50

history = model.fit(segments,
                      labels_hot,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)

df_test['x'] = df_test['x'] / df_test['x'].max()
df_test['y'] = df_test['y'] / df_test['y'].max()
df_test['z'] = df_test['z'] / df_test['z'].max()

df_test = df_test.round({'x': 4, 'y': 4, 'z': 4})

segments_test, labels_test = segment_and_labels(df_test, TIME_PERIODS, STEP_VAL)

print segments_test.shape, labels_test.shape

num_time_periods, num_sensors = segments_test.shape[1], segments_test.shape[2]
num_classes = le.classes_.size

segments_test = segments_test.reshape(segments_test.shape[0], num_time_periods * num_sensors)

segments_test = segments_test.astype('float32')
labels_test = labels_test.astype('float32')

labels_hot_test = keras.utils.np_utils.to_categorical(labels_test, num_classes)

# labels_pred = model.predict(segments_test)

scores = model.evaluate(segments_test, labels_hot_test, batch_size=BATCH_SIZE, verbose=1)
print model.metrics_names
print scores
# max_y_pred_test = np.argmax(labels_pred, axis=1)
# max_y_test = np.argmax(y_test, axis=1)
