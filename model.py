from tensorflow.keras import models, layers
import numpy as np
import sincnet
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import MaxPooling1D, Conv1D, LeakyReLU, BatchNormalization, Dense, Flatten
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.models import Model

from conf import *


# added uniques names to every layer to avoid error while saving model
def get_model(input_shape, out_dim):
    inputs = Input(input_shape)
    x = sincnet.SincConv1D(cnn_N_filt[0], cnn_len_filt[0], fs)(inputs)

    x = MaxPooling1D(pool_size=cnn_max_pool_len[0], name='mx1')(x)
    if cnn_use_batchnorm[0]:
        x = BatchNormalization(momentum=0.05, name='bn1')(x)
    if cnn_use_laynorm[0]:
        x = sincnet.LayerNorm()(x)
    x = LeakyReLU(alpha=0.2, name='lr1')(x)

    x = Conv1D(cnn_N_filt[1], cnn_len_filt[1], strides=1, padding='valid', name='conv1')(x)
    x = MaxPooling1D(pool_size=cnn_max_pool_len[1], name='mx2')(x)
    if cnn_use_batchnorm[1]:
        x = BatchNormalization(momentum=0.05, name='bn2')(x)
    if cnn_use_laynorm[1]:
        x = sincnet.LayerNorm()(x)
    x = LeakyReLU(alpha=0.2, name='lr2')(x)

    x = Conv1D(cnn_N_filt[2], cnn_len_filt[2], strides=1, padding='valid', name='conv2')(x)
    x = MaxPooling1D(pool_size=cnn_max_pool_len[2], name='mx3')(x)
    if cnn_use_batchnorm[2]:
        x = BatchNormalization(momentum=0.05, name='bn3')(x)
    if cnn_use_laynorm[2]:
        x = sincnet.LayerNorm()(x)
    x = LeakyReLU(alpha=0.2, name='lr3')(x)
    x = Flatten()(x)

    # DNN
    x = Dense(fc_lay[0], name='fc1')(x)
    if fc_use_batchnorm[0]:
        x = BatchNormalization(momentum=0.05, epsilon=1e-5, name='bn4')(x)
    if fc_use_laynorm[0]:
        x = sincnet.LayerNorm()(x)
    x = LeakyReLU(alpha=0.2, name='lr4')(x)

    x = Dense(fc_lay[1], name='fc2')(x)
    if fc_use_batchnorm[1]:
        x = BatchNormalization(momentum=0.05, epsilon=1e-5, name='bn5')(x)
    if fc_use_laynorm[1]:
        x = sincnet.LayerNorm()(x)
    x = LeakyReLU(alpha=0.2, name='lr5')(x)

    x = Dense(fc_lay[2], name='fc3')(x)
    if fc_use_batchnorm[2]:
        x = BatchNormalization(momentum=0.05, epsilon=1e-5, name='bn6')(x)
    if fc_use_laynorm[2]:
        x = sincnet.LayerNorm()(x)
    x = LeakyReLU(alpha=0.2, name='lr6')(x)

    # DNN final
    prediction = layers.Dense(out_dim, activation='softmax', name='fc4')(x)
    model = Model(inputs=inputs, outputs=prediction, name='SincNet_model')
    model.summary()
    return model
