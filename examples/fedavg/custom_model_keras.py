import os
import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

import logging
logger = logging.getLogger(__name__)


def get_hyperparams():
    local_params = {
        'training': {
            'epochs': 1,
            'batch_size': 1,
            'steps_per_epoch': 1,
        }
    }
    
    return local_params

def get_model_config(folder_configs, dataset, is_agg=False, party_id=0):

    if is_agg:
        return None

    nn_hidden_layers_num = 100
    nn_params_per_layer = 100

    print("Housing MLP specs\n" + "Hidden Layers:{}\n".format(nn_hidden_layers_num) + "Params Per Layer:{}\n".format(nn_params_per_layer))

    model = HousingMLP(params_per_layer=nn_params_per_layer, hidden_layers_num=nn_hidden_layers_num).get_model()

    if not os.path.exists(folder_configs):
        os.makedirs(folder_configs)

    # Save model
    fname = os.path.join(folder_configs, 'compiled_keras.h5')
    model.save(fname)

    K.clear_session()
    # Generate model spec:
    spec = {
        'model_name': 'keras-cnn',
        'model_definition': fname
    }

    model = {
        'name': 'KerasFLModel',
        'path': 'ibmfl.model.keras_fl_model',
        'spec': spec
    }

    return model



class HousingMLP:
    def __init__(self, params_per_layer=10, hidden_layers_num=1, learning_rate=0.0, data_type="float32"):
        super(HousingMLP, self).__init__()
        self.params_per_layer = params_per_layer
        self.hidden_layers_num = hidden_layers_num
        # We set the default value to 0.0 so not to train,
        # since this model is solely used for stress testing.
        self.learning_rate = learning_rate

        if data_type == "float32":
            self.data_type = tf.float32
        elif data_type == "float64":
            self.data_type = tf.float64
        else:
            raise RuntimeError("Not a supported data type. Please pass float32 or float64")

    def get_model(self):
        model = tf.keras.models.Sequential()
        # This layer outputs 14x10, 14x100, 14x1000, etc...
        model.add(tf.keras.layers.Dense(self.params_per_layer,
                                        input_shape=(13,),
                                        kernel_initializer='normal',
                                        activation='relu',
                                        dtype=self.data_type))
        for i in range(self.hidden_layers_num):
            model.add(tf.keras.layers.Dense(self.params_per_layer,
                                            input_shape=(self.params_per_layer,),
                                            kernel_initializer='normal',
                                            activation='relu',
                                            dtype=self.data_type))
        model.add(tf.keras.layers.Dense(1,
                                        kernel_initializer='normal',
                                        dtype=self.data_type))
        # We set a very
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model