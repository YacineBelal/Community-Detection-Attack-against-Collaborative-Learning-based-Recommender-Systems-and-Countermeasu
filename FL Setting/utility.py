import os
import random
import numpy as np
from configparser import ConfigParser
import sys 
# may put this function in another utility file
def import_tensorflow():
    # Filter tensorflow version warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)                        

    import logging
    tf.get_logger().setLevel(logging.ERROR)
    return tf

tf = import_tensorflow()    

from tensorflow import keras 
from tensorflow.keras.layers import Input,Embedding, Flatten, Multiply, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal


cost_file_name = "overhead.ini"

def reset_random_seeds():
    os.environ['PYTHONHASHSEED'] = str(1)
    np.random.seed(1)
    random.seed(1)

reset_random_seeds()


class GMF(keras.Model):
    def train_step(self, data):
        x , y = data
        self.sensitivity = 1
        self.noise_mult = 4.84
        self.delta = 10e-6

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Clip the gradients to enforce privacy
        clipped_gradients = [tf.clip_by_value(gradient, -self.sensitivity, self.sensitivity) for gradient in gradients]

        noisy_gradients = []
        for gradient in clipped_gradients:
            noise = tf.random.normal(gradient.shape, mean = 0.0, stddev = self.noise_mult)
            noisy_gradients.append(gradient + noise)
            # self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon) #self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon)
        
        # Update the model parameters using the noisy gradients
        self.optimizer.apply_gradients(zip(noisy_gradients, trainable_vars))

        # self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

def get_model(num_items, num_users):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    MF_Embedding_User = Embedding(input_dim=num_users, output_dim=8, name='user_embedding',
                                  embeddings_initializer=RandomNormal, embeddings_regularizer=l2(0), input_length=1)

    MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=8, name='item_embedding',
                                  embeddings_initializer=RandomNormal, embeddings_regularizer=l2(0), input_length=1)

    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))

    # Element-wise product of user and item embeddings 
    predict_vector = Multiply()([user_latent, item_latent])
    # dropout_1 = Dropout(0.2,name='Dropout_1')(predict_vector)

    # dense_1 = Dense(40,name='FullyConnected-2')(dropout_1)
    # dropout_2 = Dropout(0.2,name='Dropout_2')(dense_1)

    # dense_2 = Dense(10,name='FullyConnected-3')(dropout_2)
    # dropout_3 = Dropout(0.2,name='Dropout_3')(dense_2)

    # Final prediction layer
    # prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(predict_vector)

    model = GMF(inputs=[user_input, item_input],
                  outputs=prediction)

    return model

def init_cost_file():
    config_object = ConfigParser()
    config_object["performance"] = {
        "Aggregation_Time_Per_Round": "0",
        "Aggregation_Time_Total": "0",
        "Transfer_Time_Per_Round_To_Server": "0",
        "Transfer_Time_Total_To_Server": "0",
        "Transfer_Time_Init_Total": "0"
    }
    with open(cost_file_name, "w") as cost_file:
        config_object.write(cost_file)

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
