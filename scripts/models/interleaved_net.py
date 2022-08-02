import os

print('importing tf')
import tensorflow
print('done')
import keras

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dropout
from keras.layers import BatchNormalization, add
from keras.layers.pooling import GlobalAvgPool2D
# from keras.layers.merge import concatenate
from keras.layers import concatenate
from keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
from sklearn import preprocessing
import keras.backend as K
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sys

from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.metrics import categorical_accuracy
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
import ast

import random
import tqdm



from tensorflow.keras.utils import set_random_seed
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# param


# learning_rate = float(sys.argv[1])
# batch_size = int(sys.argv[2])
# decay_rate = float(sys.argv[3])
# l2_value = float(sys.argv[4])
# epoch = int(sys.argv[5])
'''
# param
learning_rate = 0.001
batch_size = 128
decay_rate = 0
l2_value = 0
epoch = 10000
'''


def get_f1(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def shuffle_train_test_split(size, ratio):
    index = np.arange(size)
    shuffle(index, random_state=0)
    sep = int(size * ratio)
    return index[:sep], index[sep:]


n_body_pose = 14 * 2
n_hands_pose = 21 * 4
n_face_pose = 70 * 2

n_gaze_pose = 1 * 2
n_head_pose = 1 * 2

n_time = 1

n_speaking = 1

feature_sizes = {'body':14*2, 'face':70*2, 'gaze':2, 'headpose':2, 'time_since_last_bite':1, 'time_since_start':1, 'num_bites':1, 'speaking':1}

gpus = tf.config.list_physical_devices('GPU')

print(gpus)

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])


tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)


class PazNet:
    def __init__(self, l2_value=0, frame_length=90, inputs=['body', 'face', 'gaze', 'headpose', 'speaking']): 
        # open pose channel
        self.inputs = inputs
        self.l2_value = l2_value
        self.frame_length = frame_length

        input_size = 0
        for inp in inputs:
            input_size += feature_sizes[inp]
        print('Input Size:', input_size)

        self.input_size = input_size

    def create_model(self):
        l2_value = self.l2_value
        frame_length = self.frame_length
        input_size = self.input_size
        pool_num = 2
        k_size = 3

        if input_size <= feature_sizes['gaze'] + feature_sizes['headpose'] + feature_sizes['speaking']:
            pool_num = 1

        if input_size <= feature_sizes['gaze'] + feature_sizes['headpose']:
            k_size = 1

        # input 1
        input1 = Input(shape=(frame_length, input_size, 1))
        bn11 = BatchNormalization()(input1)
        conv11 = Conv2D(32, k_size, padding='same', kernel_initializer="he_normal", activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            bn11)
        bn12 = BatchNormalization()(conv11)
        conv12 = Conv2D(32, k_size, padding='same', kernel_initializer="he_normal",activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            bn12)
        bn13 = BatchNormalization()(conv12)

        conv13 = Conv2D(32, k_size, padding='same', kernel_initializer="he_normal",activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            bn13)
        bn14 = BatchNormalization()(conv13)

        conv14 = Conv2D(16, k_size, strides=2, kernel_initializer="he_normal", activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            bn14)
        pool11 = MaxPooling2D(pool_size=(pool_num, pool_num))(conv14)
        bn15 = BatchNormalization()(pool11)
        conv15 = Conv2D(8, kernel_size=1, strides=2, kernel_initializer="he_normal", activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            bn15)
        pool12 = MaxPooling2D(pool_size=(pool_num, pool_num))(conv15)
        bn16 = BatchNormalization()(pool12)
        flat1 = Flatten()(bn16)

        # input 2
        input2 = Input(shape=(frame_length, input_size, 1))
        bn21 = BatchNormalization()(input2)
        add21 = add([bn12, bn14, bn21])
        conv21 = Conv2D(32, k_size, padding='same', kernel_initializer="he_normal",activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            add21)
        bn22 = BatchNormalization()(conv21)
        add22 = add([bn13, bn22])
        conv22 = Conv2D(32, k_size, padding='same', kernel_initializer="he_normal",activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            add22)
        bn23 = BatchNormalization()(conv22)
        add23 = add([bn23, bn14])
        conv23 = Conv2D(32, k_size, padding='same', kernel_initializer="he_normal",activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            add23)
        bn24 = BatchNormalization()(conv23)

        conv24 = Conv2D(16, k_size, strides=2, kernel_initializer="he_normal", activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            bn24)
        pool21 = MaxPooling2D(pool_size=(pool_num, pool_num))(bn24)
        bn25 = BatchNormalization()(pool21)
        conv25 = Conv2D(8, kernel_size=1, strides=2, kernel_initializer="he_normal", activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            bn25)
        pool22 = MaxPooling2D(pool_size=(pool_num, pool_num))(conv25)
        bn26 = BatchNormalization()(pool22)
        flat2 = Flatten()(bn26)

        # input 3
        input3 = Input(shape=(frame_length, input_size, 1))
        bn31 = BatchNormalization()(input3)
        add31 = add([bn22, bn24, bn31])
        conv31 = Conv2D(32, k_size, padding='same', kernel_initializer="he_normal",activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            add31)
        bn32 = BatchNormalization()(conv31)
        add32 = add([bn23, bn32])
        conv32 = Conv2D(32, k_size, padding='same', kernel_initializer="he_normal",activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            add32)
        bn33 = BatchNormalization()(conv32)
        add33 = add([bn33, bn24])
        conv33 = Conv2D(32, k_size, padding='same', kernel_initializer="he_normal",activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            add33)
        bn34 = BatchNormalization()(conv33)
        conv34 = Conv2D(16, k_size, strides=2, kernel_initializer="he_normal", activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            bn34)
        pool31 = MaxPooling2D(pool_size=(pool_num, pool_num))(conv34)
        bn35 = BatchNormalization()(pool31)
        conv35 = Conv2D(16, kernel_size=1, strides=2, kernel_initializer="he_normal", activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(
            bn35)
        pool32 = MaxPooling2D(pool_size=(pool_num, pool_num))(bn35)
        bn36 = BatchNormalization()(pool32)
        flat3 = Flatten()(bn36)

        # merge them together
        merge = concatenate([flat1, flat2, flat3])

        # get output
        hidden1 = Dense(32, activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(merge)
        dropout1 = Dropout(0.05)(hidden1)
        hidden2 = Dense(8, activation='relu', kernel_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(dropout1)
        dropout2 = Dropout(0.05)(hidden2)
        bn5 = BatchNormalization()(dropout2)

        output = Dense(1, activation='sigmoid')(bn5)


        self.model = Model([input1, input2, input3], output)
        self.model.summary()

        # tf.keras.utils.plot_model(self.model, to_file='model.png', show_shapes=True)

        self.metrics = [
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
        ]

    def load(self, path="/home/abrar/feeding_ws/src/bite-timing-robot/weights/interleaved_net_6s_he_nohand_15fps_0.001_0.0_0.005_32.h5"):
        self.model = keras.models.load_model(path,
                                                custom_objects={'get_f1': get_f1})
        self.predict_layer = Model(inputs=self.model.input, outputs=self.model.output)


    def predict(self, persons):


        predict_layer_output = self.predict_layer.predict(x=persons)
        print(predict_layer_output)

        predicted_y = np.argmax(predict_layer_output, axis=1)
        # predicted_y = pd.get_dummies(predicted_y)
        print(predicted_y)
        return predicted_y

    '''
        learning_rate = 0.001
        batch_size = 128
        decay_rate = 0
        l2_value = 0
        epoch = 10000
    '''

def generator(main,left,right,labels, batch_size): # Create empty arrays to contain batch of features and labels# batch_features = np.zeros((batch_size, 64, 64, 3))
    batch_labels = np.zeros((batch_size,1)) 
    p1 = np.zeros((batch_size, main.shape[1], main.shape[2]))
    p2 = np.zeros((batch_size, main.shape[1], main.shape[2]))
    p3 = np.zeros((batch_size, main.shape[1], main.shape[2]))
    cur = 0

    idxs = np.arange(len(main))
    np.random.shuffle(idxs)
    # set random shuffle
    main = main[idxs]
    left = left[idxs]
    right = right[idxs]
    labels = labels[idxs]

    while True:
        for i in range(batch_size):
            # choose random index in features
            if cur == len(labels):
                cur = 0
                # and random shuffle
                idxs = np.random.shuffle(idxs)
                main = main[idxs]
                left = left[idxs]
                right = right[idxs]
                labels = labels[idxs]

            p1[i] = main[cur]
            p2[i] = left[cur]
            p3[i] = right[cur]
            batch_labels[i] = labels[cur]


        yield [p1, p2, p3], batch_labels

def train(args):

    from data_utils import organize_data


    features = args.features
    global_features = args.global_features
    epoch = args.epoch
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    l2_value = args.l2_value
    patience = args.patience
    decay_rate = args.decay_rate
    seed = args.seed
    training_split = args.training_split
    use_ssp = args.use_ssp
    frame_length = args.frame_length


    # set seeds
    np.random.seed(seed)
    set_random_seed(seed)

    print('Loading data...')
    main, left, right, global_feats, labels, ids = organize_data(features, global_features, use_ssp)
    y = labels
    print('Loaded data.')

    # normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    X_norm = min_max_scaler.fit_transform(main.reshape(-1, main.shape[1]*main.shape[2]))
    X_norm = X_norm.reshape(main.shape)

    min_max_scaler = preprocessing.MinMaxScaler()
    X2_norm = min_max_scaler.fit_transform(left.reshape(-1, left.shape[1]*left.shape[2]))
    X2_norm = X2_norm.reshape(left.shape)

    min_max_scaler = preprocessing.MinMaxScaler()
    X3_norm = min_max_scaler.fit_transform(right.reshape(-1, right.shape[1]*right.shape[2]))
    X3_norm = X3_norm.reshape(right.shape)

    X = np.stack((X_norm, X2_norm, X3_norm), axis=1)


    iterate = []

    # training split types: 70:30, 10-fold, loso-subject, loso-session
    if training_split == "70:30":
        iterate = [0]
    elif training_split == "10-fold":
        iterate = [0,1,2,3,4,5,6,7,8,9]
    elif training_split == "loso-subject":
        iterate = np.array(list(sorted(set(ids))))
    elif training_split == "loso-session":
        sessions = set()
        for i in range(len(ids)):
            sessions.add(ids[i][:2])
        iterate = np.array(sorted(list(sessions)))

    # only for 10-fold
    kfold = KFold(n_splits=10, shuffle=True)
    tenFoldSplits = list(kfold.split(X, y))

    for i in tqdm.tqdm(iterate):
        print("Training split is: ", training_split)
        print("Training fold / subject / session is: ", i)


        # shuffle and split
        # train_index, test_index = shuffle_train_test_split(len(y), 0.8)
        print("Making test-train splits")

        if training_split == "70:30":
            x_ids = list(range(len(X)))
            X_train_ids, X_test_ids, y_train, y_test = train_test_split(x_ids, y, test_size=0.3)

        elif training_split == "10-fold":
            # the value of iterate[i] is the fold number
            X_train_ids, X_test_ids = tenFoldSplits[i]
            y_train = y[X_train_ids]
            y_test = y[X_test_ids]

        elif training_split == "loso-subject":
            # the value of iterate[i] is the subject id for the test set!
            X_train_ids = []
            X_test_ids = []
            for j in range(len(y)):
                if ids[j] == i:
                    X_test_ids.append(j)
                else:
                    X_train_ids.append(j)
            y_train = y[X_train_ids]
            y_test = y[X_test_ids]

        elif training_split == "loso-session":
            # the value of iterate[i] is the session id for the test set!
            X_train_ids = []
            X_test_ids = []
            for j in range(len(y)):
                if ids[j][:2] == i:
                    X_test_ids.append(j)
                else:
                    X_train_ids.append(j)

            y_train = y[X_train_ids]
            y_test = y[X_test_ids]
        else:
            print("Error: training split not recognized")
            exit()



        print("Done making test-train splits")


        # split people up

        X1_train = X[X_train_ids, 0, :]
        X2_train = X[X_train_ids, 1, :]
        X3_train = X[X_train_ids, 2, :]
        X1_test = X[X_test_ids, 0, :]
        X2_test = X[X_test_ids, 1, :]
        X3_test = X[X_test_ids, 2, :]


        print('Checking Assertions')
        assert not np.any(np.isnan(X1_train))
        assert not np.any(np.isnan(X2_train))
        assert not np.any(np.isnan(X3_train))
        assert not np.any(np.isnan(X1_test))
        assert not np.any(np.isnan(X2_test))
        assert not np.any(np.isnan(X3_test))

        assert not np.any(np.isnan(y_train))
        assert not np.any(np.isnan(y_test))
        print("Assertions Valid")

        print("Training")
        print(X1_train.shape)
        print(X2_train.shape)
        print(X3_train.shape)
        print(y_train.shape)
        print('Number of positive samples in training: ',np.sum(y_train))
        print('Number of negative samples in training: ',len(y_train)-np.sum(y_train))

        print("Test")
        print(X1_test.shape)
        print(X2_test.shape)
        print(X3_test.shape)
        print(y_test.shape)
        print('Number of positive samples in test: ', np.sum(y_test))
        print('Number of negative samples in test: ', len(y_test) - np.sum(y_test))


        # create paznet
        paznet = PazNet(inputs=features, frame_length=180)
        paznet.create_model()

        import wandb
        from wandb.keras import WandbCallback
        feats = '_'.join(sorted(features))
        feats = 'interleaved_paznet' + '_' + feats

        config = {'training_split':training_split, 'test_split_value':i}
        config.update(vars(args))

        split_string = training_split + '_' + str(i)
        if use_ssp:
            split_string += '_ssp'
            training_split += '_ssp'

        print("Creating wandb")
        run = wandb.init(project='social-dining', group=feats + '_' + training_split, config=config, name=feats+ '_' + split_string)
        print("Wandb Run: ", run)


        print("Creating lr scheduler")
        # learning rate decay
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=1000,
            decay_rate=decay_rate)

        # early stopping
        print('Creating model callbacks')
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)

        # save the best model by measuring F1-score
        mc = ModelCheckpoint("checkpoints/" + feats + '_' + split_string + '_' + str(
            learning_rate) + "_" + str(decay_rate) + "_" + str(l2_value) + '_' + str(batch_size) + ".h5",
                            monitor='val_get_f1', mode='max', verbose=1, save_best_only=True)

        print('Compiling model')
        paznet.model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=binary_crossentropy, metrics=[get_f1, paznet.metrics])

        print('Starting training')
        history = paznet.model.fit(x=[X1_train, X2_train, X3_train], y=y_train, epochs=epoch,
                            batch_size=batch_size, validation_data=([X1_test, X2_test, X3_test], y_test), callbacks=[es, mc, WandbCallback()])
        
        # history = paznet.model.fit_generator(generator(X1_train,X2_train,X3_train,y_train, batch_size), epochs=epoch,
        #                     validation_data=generator(X1_test,X2_test,X3_test,y_test, batch_size), callbacks=[es, mc, WandbCallback()])
        wandb.finish()

if __name__ == '__main__':
    # arg parse the features lists
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', nargs='+', type=str, default=['body', 'face', 'gaze', 'headpose', 'speaking'],
                        help='features to use')
    parser.add_argument('--global_features', nargs='+', type=str, default=['num_bites', 'time_since_last_bite']
                        , help='global features to use')
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--l2_value', type=float, default=0.0001, help='l2 regularization value')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')
    parser.add_argument('--decay_rate', type=float, default=0.5, help='learning rate decay rate')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--training_split', type=str, default='70:30', help='training split')
    parser.add_argument('--use_ssp', type=int, default=0, help='whether to use social signal processing-like method')
    parser.add_argument('--frame_length', type=int, default=180, help='frames to sample in a 6s sample')

    args = parser.parse_args()

    # train()
    train(args)