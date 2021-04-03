#libraries
import pandas as pd 
from sklearn import preprocessing
from sklearn.utils import class_weight
import numpy as np 
import pickle
from sklearn.preprocessing import OneHotEncoder 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
import math
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import LeakyReLU, Dense, Dropout, BatchNormalization, Conv1D, Flatten, Input, MaxPooling1D, Add
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow import keras
import os
import random
from imblearn.over_sampling import SMOTE

# Apparently you may use different seed values at each stage
seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# GPU config for Vamsi's Laptop
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

tf.keras.backend.clear_session()
config = ConfigProto()
config.gpu_options.allow_growth = True
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

LIMIT = 3 * 1024
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=LIMIT)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

# dataset import 
ds = pd.read_csv('../../processed_data/seq_temp_all.csv')
ds = ds[ds['sequences'].notna()] # 20 sequences not available
ds.to_csv('processed_temp_all.csv')

X = np.load('../../processed_data/PB_all.npz')['arr_0']
X = np.array(X, dtype = 'f')
y = list(ds["temperatures"])
y = np.array(y, dtype='f')

# threshold 55
thresh = 55.0
y_55 = []

for i in y:
    if i >= thresh:
        y_55.append(1)
    else:
        y_55.append(0)

y_55 = np.asarray(y_55)

unique, counts = np.unique(y_55, return_counts=True)
print(unique, counts)

# Non-SMOTE
# y_55 = to_categorical(y_55)
# X = np.expand_dims(X, axis = 1)

# y process
print("Loaded X and y")

X, y = shuffle(X, y_55, random_state=42)
print("Shuffled")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.125, random_state = 42) 

# # SMOTE
# smote = SMOTE(random_state = 101)
# X_train, y_train = smote.fit_resample(X_train, y_train)
# X_val, y_val = smote.fit_resample(X_val, y_val)

X_train = np.expand_dims(X_train, axis = 1)
X_val = np.expand_dims(X_val, axis = 1)
X_test = np.expand_dims(X_test, axis = 1)

class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
print(class_weights)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

print(X_test.shape, X_train.shape, X_val.shape, y_test.shape, y_train.shape, y_val.shape)
print("Conducted Train-Test Split")

# parameters
bs = 256
num_epochs = 300
weights = {0:0.62277937, 1:2.5361727}
dropout_val = 0.5
model_file_name = '../saved_models/cnn_pb_all_clas_55.h5'

def ResBlock(inp):
    x = Conv1D(1024, (3), kernel_initializer = 'glorot_uniform', padding="same")(inp)
    x = LeakyReLU(alpha = 0.05)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_val)(x)
    x = Conv1D(1024, (3), kernel_initializer = 'glorot_uniform', padding="same")(x)
    x = LeakyReLU(alpha = 0.05)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_val)(x)
    # x = Add()([x, inp])

    return x

# keras nn model
input_ = Input(shape = (1,1024,))
x = Conv1D(128, (3), padding="same", activation = "relu")(input_)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)
x = Conv1D(128, (3), padding="same", activation = "relu")(x)
x = Dropout(0.1)(x)
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dense(256, activation = "relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)
x = Dense(256, activation = "relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)
x = Dense(256, activation = "relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.1)(x) 
out = Dense(2, activation = 'softmax')(x)
model = Model(input_, out)

print(model.summary())

# adam optimizer
opt = keras.optimizers.Adam(learning_rate = 1e-4)

# sensitivity metric
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics=['accuracy'])

# callbacks
mcp_save = keras.callbacks.ModelCheckpoint(model_file_name, save_best_only=True, monitor='val_accuracy', verbose=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_accuracy', factor=0.1, patience=30, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [reduce_lr, mcp_save]

# training

with tf.device('/gpu:0'): # use gpu
    history = model.fit(X_train, y_train, batch_size = bs, epochs = num_epochs, validation_data = (X_val, y_val), shuffle = False, callbacks = callbacks_list, class_weight = weights)
    
    # Plot History
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    model = load_model(model_file_name, custom_objects = {'sensitivity': sensitivity})
    
    # Validation Set
    y_pred = model.predict(X_val)
    print("Classification Report Validation")
    print(classification_report(y_val.argmax(axis=1), y_pred.argmax(axis=1)))
    print("Confusion Matrix")
    matrix = confusion_matrix(y_val.argmax(axis=1), y_pred.argmax(axis=1))
    print(matrix)
    print("F1 Score")
    print(f1_score(y_val.argmax(axis=1), y_pred.argmax(axis=1), average = 'weighted'))

    # Test Set
    y_pred = model.predict(X_test)
    print("Classification Report Test")
    print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
    print("Confusion Matrix")
    matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(matrix)
    print("F1 Score")
    print(f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average = 'weighted'))

    # PETase Set
    petase_pb = np.load('../../processed_data/PB_PETase.npz')['arr_0']
    petase_pb = np.expand_dims(petase_pb, axis=1)

    y_pred = model.predict(petase_pb).argmax(axis=1)
    y_true = [0,1,1,0,1,1]
    print("\nPETase Predictions \n")
    print(y_true, y_pred)
    print("Classification Report")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix")
    matrix = confusion_matrix(y_true, y_pred)
    print(matrix)
    print("F1 Score")
    print(f1_score(y_true, y_pred, average = 'weighted'))

'''
Classification Method:

Classes: 
0: Less than 70 degrees
1: >= 70

ProtBert Embeddings + Convolutional Neural Network

Classification Report
              precision    recall  f1-score   support

           0       0.97      0.94      0.95      1117
           1       0.57      0.70      0.63       126

    accuracy                           0.92      1243
   macro avg       0.77      0.82      0.79      1243
weighted avg       0.92      0.92      0.92      1243

Confusion Matrix
[[1050   67]
 [  38   88]]
F1 Score
0.9193303887978407
'''


'''SMOTE Sampling:
Threshold 70
Accuracy: 0.97558
Classification Report: BS 32
              precision    recall  f1-score   support

           0       1.00      0.95      0.97      1121
           1       0.96      1.00      0.98      1131

    accuracy                           0.98      2252
   macro avg       0.98      0.98      0.98      2252
weighted avg       0.98      0.98      0.98      2252

Confusion Matrix
[[1069   52]
 [   3 1128]]
F1 Score
0.9755633329631406
'''
'''
Brenda Dataset + ProtBert + SMOTE + Convolutional Neural Network 
'''