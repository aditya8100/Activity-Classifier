import matplotlib.pyplot as plt
import keras
import numpy as np
import pandas as pd
from IPython.display import display, HTML
from scipy import stats

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing


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


def segment_and_labels (dframe, time_step, step, label):
    N_FEATURES = 3

    segments = []
    labels = []

    for i in range(0, len(dframe) - time_step, step):
        xs = dframe['x'].values[i:i+time_step]
        ys = dframe['y'].values[i:i+time_step]
        zs = dframe['z'].values[i:i+time_step]

        label = stats.mode(dframe[label][i:i+time_step])[0][0]
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
plt.show()

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

segments, labels = segment_and_labels(df_train, TIME_PERIODS, STEP_VAL, LABEL)

print segments.shape

