import os

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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sys

from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.metrics import categorical_accuracy
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
import ast

from tcn import TCN

import argparse
from tensorflow.keras.utils import set_random_seed
from sklearn.model_selection import KFold
import random
import tqdm
import joblib

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
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


# read data
'''
time_series = np.load("data/main_np.npy", allow_pickle=True)
subject2 = np.load("data/dinner1_np.npy", allow_pickle=True)
subject3 = np.load("data/dinner2_np.npy", allow_pickle=True)
'''
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


class TCNModels:
    def __init__(self, l2_value=0, frame_length=90, inputs=['body', 'face', 'gaze', 'headpose', 'speaking', 'time_since_last_bite', 'num_bites']): 
        # open pose channel
        self.inputs = inputs
        self.l2_value = l2_value
        self.frame_length = frame_length

        input_size = 0
        video_inputs = 0
        audio_inputs = 0
        time_inputs = 0
        count_inputs = 0
        for inp in inputs:
            input_size += feature_sizes[inp]
            # video input size
            if inp in ['body', 'face', 'gaze', 'headpose']:
                video_inputs += feature_sizes[inp]
            # audio input size
            if inp in ['speaking']:
                audio_inputs += feature_sizes[inp]
            # time input size
            if inp in ['time_since_last_bite', 'time_since_start']:
                time_inputs += feature_sizes[inp]
            # count input size
            if inp in ['num_bites']:
                count_inputs += feature_sizes[inp]

        print('Input Size:', input_size)
        print('Video Input Size:', video_inputs)
        print('Audio Input Size:', audio_inputs)
        print('Time Input Size:', time_inputs)
        print('Count Input Size:', count_inputs)

        self.input_size = input_size
        self.video_input_size = video_inputs
        self.audio_input_size = audio_inputs
        self.time_input_size = time_inputs
        self.count_input_size = count_inputs

    def create_model(self, model='global'):
        l2_value = self.l2_value
        frame_length = self.frame_length
        input_size = self.input_size
        # input 1
        video_inputs = [Input(shape=(self.frame_length, self.video_input_size)) for _ in range(3)]
        
        # input 2
        audio_inputs = [Input(shape=(self.frame_length, self.audio_input_size)) for _ in range(3)]


        time_input = Input(shape=(self.time_input_size), name='time_input')
        count_input = Input(shape=(self.count_input_size), name='count_input')

        self.tf_inputs = [
            audio_inputs[0], video_inputs[0],
            audio_inputs[1], video_inputs[1],
            audio_inputs[2], video_inputs[2],
            time_input,
            count_input
        ]

        if model=='global':
            self.model = self.global_tcn(audio_inputs, video_inputs, time_input, count_input)
        elif model=='per_modality':
            self.model = self.per_modality_tcn(audio_inputs, video_inputs, time_input, count_input)
        elif model=='per_modality_seq':
            self.model = self.per_modality_sequence_tcn(audio_inputs, video_inputs, time_input, count_input)


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




    def global_tcn(self, audio_inputs, video_inputs, time_input, count_input):

        x = concatenate([
            audio_inputs[0], video_inputs[0],
            audio_inputs[1], video_inputs[1],
            audio_inputs[2], video_inputs[2],
        ])
        # x = [
        #     audio_inputs[0], video_inputs[0],
        #     audio_inputs[1], video_inputs[1],
        #     audio_inputs[2], video_inputs[2],
        # ]

        tcn_global = TCN(
            nb_filters=100,             # The number of filters to use in the convolutional layers. Would be similar to units for LSTM. Can be a list.
            kernel_size=7,              # The size of the kernel to use in each convolutional layer.
            nb_stacks=1,                # The number of stacks of residual blocks to use.
            dilations=(1, 2, 4, 8),     # We want window_size (180) =< receptive_field (181) = 1 + 2*(kernel_size - 1)*nb_stacks*sum(dilations)
            padding='causal',           # Keep causal for a causal network
            use_skip_connections=True,
            dropout_rate=0.05,
            return_sequences=False,
            activation='relu',
            kernel_initializer='he_normal',
            use_batch_norm=False,
            use_layer_norm=False,
            use_weight_norm=False,
            name='TCN'
        )
        x = concatenate([ tcn_global(x), time_input, count_input ])

        outputs = Dense(1, activation='sigmoid', name='food_lifted')(x)
        model = Model(self.tf_inputs, outputs, name='TCN_global_model')

        return model


    def per_modality_tcn(self, audio_inputs, video_inputs, time_input, count_input):
        x_audio = concatenate([audio_inputs[0], audio_inputs[1], audio_inputs[2]])
        x_video = concatenate([video_inputs[0], video_inputs[1], video_inputs[2]])

        tcn_audio = TCN(
            nb_filters=50,             # The number of filters to use in the convolutional layers. Would be similar to units for LSTM. Can be a list.
            kernel_size=7,              # The size of the kernel to use in each convolutional layer.
            nb_stacks=1,                # The number of stacks of residual blocks to use.
            dilations=(1, 2, 4, 8),     # We want window_size (180) =< receptive_field (181) = 1 + 2*(kernel_size - 1)*nb_stacks*sum(dilations)
            padding='causal',           # Keep causal for a causal network
            use_skip_connections=True,
            dropout_rate=0.05,
            return_sequences=False,
            activation='relu',
            kernel_initializer='he_normal',
            use_batch_norm=False,
            use_layer_norm=False,
            use_weight_norm=False,
            name='tcn_audio'
        )
        tcn_video = TCN(
            nb_filters=100,             # The number of filters to use in the convolutional layers. Would be similar to units for LSTM. Can be a list.
            kernel_size=7,              # The size of the kernel to use in each convolutional layer.
            nb_stacks=1,                # The number of stacks of residual blocks to use.
            dilations=(1, 2, 4, 8),     # We want window_size (180) =< receptive_field (181) = 1 + 2*(kernel_size - 1)*nb_stacks*sum(dilations)
            padding='causal',           # Keep causal for a causal network
            use_skip_connections=True,
            dropout_rate=0.05,
            return_sequences=False,
            activation='relu',
            kernel_initializer='he_normal',
            use_batch_norm=False,
            use_layer_norm=False,
            use_weight_norm=False,
            name='tcn_video'
        )
        x = concatenate([ tcn_audio(x_audio), tcn_video(x_video), time_input, count_input ])

        outputs = Dense(1, activation='sigmoid', name='food_lifted')(x)
        model = Model(self.tf_inputs, outputs, name='TCN_per-modality_model')
        return model

    def per_modality_sequence_tcn(self, audio_inputs, video_inputs, time_input, count_input):
        x_audio = concatenate([audio_inputs[0], audio_inputs[1], audio_inputs[2]])
        x_video = concatenate([video_inputs[0], video_inputs[1], video_inputs[2]])

        tcn_audio = TCN(
            nb_filters=50,             # The number of filters to use in the convolutional layers. Would be similar to units for LSTM. Can be a list.
            kernel_size=7,              # The size of the kernel to use in each convolutional layer.
            nb_stacks=1,                # The number of stacks of residual blocks to use.
            dilations=(1, 2, 4, 8),     # We want window_size (180) =< receptive_field (181) = 1 + 2*(kernel_size - 1)*nb_stacks*sum(dilations)
            padding='causal',           # Keep causal for a causal network
            use_skip_connections=True,
            dropout_rate=0.05,
            return_sequences=True,
            activation='relu',
            kernel_initializer='he_normal',
            use_batch_norm=False,
            use_layer_norm=False,
            use_weight_norm=False,
            name='tcn_audio'
        )
        tcn_video = TCN(
            nb_filters=100,             # The number of filters to use in the convolutional layers. Would be similar to units for LSTM. Can be a list.
            kernel_size=7,              # The size of the kernel to use in each convolutional layer.
            nb_stacks=1,                # The number of stacks of residual blocks to use.
            dilations=(1, 2, 4, 8),     # We want window_size (180) =< receptive_field (181) = 1 + 2*(kernel_size - 1)*nb_stacks*sum(dilations)
            padding='causal',           # Keep causal for a causal network
            use_skip_connections=True,
            dropout_rate=0.05,
            return_sequences=True,
            activation='relu',
            kernel_initializer='he_normal',
            use_batch_norm=False,
            use_layer_norm=False,
            use_weight_norm=False,
            name='tcn_video'
        )
        tcn_audiovideo = TCN(
            nb_filters=50,              # The number of filters to use in the convolutional layers. Would be similar to units for LSTM. Can be a list.
            kernel_size=7,              # The size of the kernel to use in each convolutional layer.
            nb_stacks=1,                # The number of stacks of residual blocks to use.
            dilations=(1, 2, 4, 8),     # We want window_size (180) =< receptive_field (181) = 1 + 2*(kernel_size - 1)*nb_stacks*sum(dilations)
            padding='causal',           # Keep causal for a causal network
            use_skip_connections=True,
            dropout_rate=0.05,
            return_sequences=False,
            activation='relu',
            kernel_initializer='he_normal',
            use_batch_norm=False,
            use_layer_norm=False,
            use_weight_norm=False,
            name='tcn_audiovideo'
        )
        x = tcn_audiovideo( concatenate([ tcn_audio(x_audio), tcn_video(x_video) ]) )
        x = concatenate([ x, time_input, count_input ])

        outputs = Dense(1, activation='sigmoid', name='food_lifted')(x)
        model = Model(self.tf_inputs, outputs, name='TCN_per-modality_sequence_model')
        return model


    def load(self, path="/home/abrar/feeding_ws/src/bite-timing-robot/weights/tcn_global_body_face_gaze_headpose_num_bites_speaking_time_since_last_bite_15fps_70:30_0_ssp_0.001_0.5_0.0001_128.h5"):
        
        used_features = []
        for feature in feature_sizes.keys():
            if feature in path:
                used_features.append(feature)
        self.model_path = path
        # making my code worse :()
        input_size = 0
        video_inputs = 0
        audio_inputs = 0
        time_inputs = 0
        count_inputs = 0
        for inp in used_features:
            input_size += feature_sizes[inp]
            # video input size
            if inp in ['body', 'face', 'gaze', 'headpose']:
                video_inputs += feature_sizes[inp]
            # audio input size
            if inp in ['speaking']:
                audio_inputs += feature_sizes[inp]
            # time input size
            if inp in ['time_since_last_bite', 'time_since_start']:
                time_inputs += feature_sizes[inp]
            # count input size
            if inp in ['num_bites']:
                count_inputs += feature_sizes[inp]

        print('Input Size:', input_size)
        print('Video Input Size:', video_inputs)
        print('Audio Input Size:', audio_inputs)
        print('Time Input Size:', time_inputs)
        print('Count Input Size:', count_inputs)

        self.input_size = input_size
        self.video_input_size = video_inputs
        self.audio_input_size = audio_inputs
        self.time_input_size = time_inputs
        self.count_input_size = count_inputs

        self.create_model('global')
        self.model.load_weights(path)
        print("Weights loaded")
        # self.model.compile()

        print("Doing 10 random forward passes")
        # do a bunch of forward passes so the delays during inference is shorter
        video_inputs = [np.random.random((1, self.frame_length, self.video_input_size)) for _ in range(3)]
        
        # input 2
        audio_inputs = [np.random.random((1, self.frame_length, self.audio_input_size)) for _ in range(3)]


        time_input = np.random.random((1, self.time_input_size))
        count_input = np.random.random((1, self.count_input_size))

        temp_input = [
            audio_inputs[0], video_inputs[0],
            audio_inputs[1], video_inputs[1],
            audio_inputs[2], video_inputs[2],
            time_input,
            count_input
        ]

        for _ in range(10):
            self.model.predict(temp_input)


        self.scalers = [None, None, None]
        print("LOADED")
        self.scalers[0] = joblib.load('/home/abrar/feeding_ws/src/bite-timing-robot/weights/tcn_global_body_face_gaze_headpose_num_bites_speaking_time_since_last_bite_15fps_70:30_0_ssp_scaler_0.pkl')
        self.scalers[1] = joblib.load('/home/abrar/feeding_ws/src/bite-timing-robot/weights/tcn_global_body_face_gaze_headpose_num_bites_speaking_time_since_last_bite_15fps_70:30_0_ssp_scaler_1.pkl')
        self.scalers[2] = joblib.load('/home/abrar/feeding_ws/src/bite-timing-robot/weights/tcn_global_body_face_gaze_headpose_num_bites_speaking_time_since_last_bite_15fps_70:30_0_ssp_scaler_2.pkl')

    def predict(self, input):

        for inp in input:
            print(inp.shape)

        if not 'no_scaling' in self.model_path:
            input[1] = self.scalers[0].transform(input[1].reshape(len(input[1]), -1))
            input[1] = input[1].reshape(len(input[1]), 90, -1)
            input[3] = self.scalers[1].transform(input[3].reshape(len(input[3]), -1))
            input[3] = input[3].reshape(len(input[3]), 90, -1)
            input[5] = self.scalers[2].transform(input[5].reshape(len(input[5]), -1))
            input[5] = input[5].reshape(len(input[5]), 90, -1)



        predict_layer_output = self.model(input, training=False)
        # print(predict_layer_output)

        # predicted_y = np.argmax(predict_layer_output, axis=1)
        predict_layer_output = np.array(predict_layer_output)
        print(predict_layer_output)
        predicted_y = predict_layer_output > 0.8
        # predicted_y = pd.get_dummies(predicted_y)
        # print(predicted_y)
        return predicted_y


audio_feature_names = ['speaking']
video_feature_names = ['gaze', 'headpose', 'body', 'face']
time_feature_names = ['time_since_last_bite', 'time_since_start']
count_feature_names = ['num_bites']

def train(args):
    from data_utils import get_feature_dicts

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



    # clean out the features and global features lists of junk
    if len(features) == 1 and features[0] == 'None':
        features = []
    if len(global_features) == 1 and global_features[0] == 'None':
        global_features = []


    any_audio = False
    any_video = False
    any_count = False
    any_time = False

    for feature in features:
        if feature in audio_feature_names:
            any_audio = True
        if feature in video_feature_names:
            any_video = True
    for feature in global_features:
        if feature in count_feature_names:
            any_count = True
        if feature in time_feature_names:
            any_time = True

    print('Loading data...')
    print(global_features)

    # # we will add elements to features and global features just in case
    # temp_features = set(features)
    # temp_features.add('speaking')
    # temp_features.add('gaze') # even though we won't necessarily use them!

    # temp_global_features = set(global_features)
    # temp_global_features.add('num_bites')
    # temp_global_features.add('time_since_last_bite')


    audio_inputs, video_inputs, time_inputs, count_inputs, labels, ids = get_feature_dicts(list(features), list(global_features), use_ssp)
    y = labels
    print('Loaded data.')

    # normalization of each feature separately. NOTE: THIS IS DIFFERENT FROM PAZNET,
    # which normalizes all features together.
    if any_video:
        video_inputs = list(video_inputs)
        for i in range(3):
            min_max_scaler = preprocessing.MinMaxScaler()
            video_inputs[i] = min_max_scaler.fit_transform(video_inputs[i].reshape(len(video_inputs[i]), -1))
            video_inputs[i] = video_inputs[i].reshape(len(video_inputs[i]), frame_length, -1)
            # # not normalizing the audio because it's only speaking labels for now
            # min_max_scaler = preprocessing.MinMaxScaler()
            # audio_inputs[i] = min_max_scaler.fit_transform(audio_inputs[i].reshape(len(audio_inputs[i]), -1))
            # audio_inputs[i] = audio_inputs[i].reshape(audio_inputs[i].shape)
            # not going to normalize time and count labels

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
    # since only the idxs matter, just putting count_inputs as the argument
    tenFoldSplits = list(kfold.split(count_inputs, y))

    for i in tqdm.tqdm(iterate):
        print("Training split is: ", training_split)
        print("Training fold / subject / session is: ", i)


        # shuffle and split
        # train_index, test_index = shuffle_train_test_split(len(y), 0.8)
        print("Making test-train splits")

        if training_split == "70:30":
            x_ids = list(range(len(y)))
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
        audio_train = [[] for _ in range(3)]
        video_train = [[] for _ in range(3)]
        time_train = []
        count_train = []

        audio_test = [[] for _ in range(3)]
        video_test = [[] for _ in range(3)]
        time_test = []
        count_test = []

        for j in X_train_ids:
            for k in range(3):
                audio_train[k].append(audio_inputs[k][j])
                video_train[k].append(video_inputs[k][j])

            time_train.append(time_inputs[j])
            count_train.append(count_inputs[j])

        for j in X_test_ids:
            for k in range(3):
                audio_test[k].append(audio_inputs[k][j])
                video_test[k].append(video_inputs[k][j])

            time_test.append(time_inputs[j])
            count_test.append(count_inputs[j])






        print('Checking Assertions')
        assert not np.any(np.isnan(y_train))
        assert not np.any(np.isnan(y_test))
        print("Assertions Valid")

        print("Training")
        print(y_train.shape)
        print('Number of positive samples in training: ',np.sum(y_train))
        print('Number of negative samples in training: ',len(y_train)-np.sum(y_train))

        print("Test")
        print(y_test.shape)
        print('Number of positive samples in test: ', np.sum(y_test))
        print('Number of negative samples in test: ', len(y_test) - np.sum(y_test))


        tcn_model = TCNModels(inputs=features + global_features, frame_length=frame_length)
        tcn_model.create_model(args.model)

        train_inp = [
                    audio_train[0], video_train[0],
                    audio_train[1], video_train[1],
                    audio_train[2], video_train[2],
                    time_train,
                    count_train
        ]

        test_inp = [
                    audio_test[0], video_test[0],
                    audio_test[1], video_test[1],
                    audio_test[2], video_test[2],
                    time_test,
                    count_test
        ]

        for j in range(len(train_inp)):
            train_inp[j] = np.array(train_inp[j]).squeeze()
            print(train_inp[j].shape)
            test_inp[j] = np.array(test_inp[j]).squeeze()
            print(test_inp[j].shape)

        # make the time and count to be the last number, so it's realistic to the real robot
        if any_time:
            train_inp[-2] = train_inp[-2][:, -1]
            test_inp[-2] = test_inp[-2][:, -1]
        if any_count:
            train_inp[-1] = train_inp[-1][:, -1]
            test_inp[-1] = test_inp[-1][:, -1]

        import wandb
        from wandb.keras import WandbCallback
        feats = '_'.join(sorted(features + global_features))
        feats = 'tcn_' + args.model + '_' + feats

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
        tcn_model.model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=binary_crossentropy, metrics=[get_f1, tcn_model.metrics])

        print('Starting training')
        history = tcn_model.model.fit(x=train_inp, y=y_train, epochs=epoch,
                            batch_size=batch_size, validation_data=(test_inp, y_test), callbacks=[es, mc, WandbCallback()])
        
        wandb.finish()

if __name__ == '__main__':
    # arg parse the features lists
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='global', help='tcn model to use')
    parser.add_argument('--features', nargs='+', type=str, default=['body', 'face', 'gaze', 'headpose', 'speaking'],
                        help='features to use')
    parser.add_argument('--global_features', nargs='+', type=str, default=['num_bites', 'time_since_last_bite']
                        , help='global features to use')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
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



# model = TCNModels()
# model.create_model('global')
# # model.create_model('per_modality')
# model.create_model('per_modality_seq')

# import numpy as np

# # expects 8 inputs
# video_inp = np.random.random((1,90,feature_sizes['body'] + feature_sizes['face'] + feature_sizes['gaze'] + feature_sizes['headpose']))
# audio_inp = np.random.random((1,90,feature_sizes['speaking']))
# count_inp = np.random.random((1))
# time_inp = np.random.random((1))

# inp = [
#             audio_inp, video_inp,
#             audio_inp, video_inp,
#             audio_inp, video_inp,
#             time_inp,
#             count_inp
#         ]


# print("Starting predict")


# import time
# start = time.time()
# out = model.predict(inp)
# end = time.time()

# print(end-start)
# print("Done", out)