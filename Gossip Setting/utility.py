import os
from configparser import ConfigParser

####*IMPORANT*: Have to do this line *before* importing tensorflow
os.environ['PYTHONHASHSEED'] = str(1)

import random
import numpy as np

import theano.tensor as T
from keras import backend as K
from keras import initializations
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Flatten, Dropout
from keras.optimizers import SGD
from keras.regularizers import l2
from Dataset import Dataset

cost_file_name = "overheadAlpha08.ini"


def reset_random_seeds():
    os.environ['PYTHONHASHSEED'] = str(1)
    np.random.seed(1)
    random.seed(1)


reset_random_seeds()


def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)

def get_model(num_items, num_users):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    MF_Embedding_User = Embedding(input_dim=num_users, output_dim=8, name='user_embedding',
                                  init=init_normal, W_regularizer=l2(0), input_length=1)

    MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=8, name='item_embedding',
                                  init=init_normal, W_regularizer=l2(0), input_length=1)

    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))

    # Element-wise product of user and item embeddings 
    predict_vector = merge([user_latent, item_latent], mode='mul')

    # prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction')(predict_vector)

    model = Model(input=[user_input, item_input],
                  output=prediction)

    return model


def init_cost_file():
    config_object = ConfigParser()
    config_object["performance"] = {
        "Aggregation_Time_Per_Round": 0,
        "Aggregation_Time_Total": 0,
        "Transfer_Time_Per_Round_To_Server": 0,
        "Transfer_Time_Total_To_Server": 0,
        "Transfer_Time_Init_Total": 0
    }
    with open(cost_file_name, "w") as cost_file:
        config_object.write(cost_file)


# init_cost_file()

def save_per_round_cost(time_t, property):
    config_object = ConfigParser()
    config_object.read(cost_file_name)
    performance = config_object["performance"]
    performance[property] = str(time_t)
    with open(cost_file_name, "w") as cost_file:
        config_object.write(cost_file)


def save_whole_cost(time_t, property):
    config_object = ConfigParser()
    config_object.read(cost_file_name)
    performance = config_object["performance"]
    if property in performance:
        performance[property] = str(float(performance[property]) + time_t)
    else:
        performance[property] = str(time_t)
    with open(cost_file_name, "w") as cost_file:
        config_object.write(cost_file)
