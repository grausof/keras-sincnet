import os
#import scipy.io.wavfile
# python test.py --cfg=cfg/SincNet_TIMIT.cfg
import soundfile as sf

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.callbacks import Callback
import sys
import numpy as np
from keras.utils import to_categorical
from data_io import ReadList,read_conf,str_to_bool
from keras.optimizers import RMSprop
from keras import backend as K
import gc
from tqdm import tqdm
from keras.layers import MaxPooling1D, Conv1D, LeakyReLU, BatchNormalization, Dense, Flatten
K.clear_session()
from test import Validation


def batchGenerator(batch_size,data_folder,wav_lst,N_snt,wlen,lab_dict,fact_amp, out_dim):
    while True:
        sig_batch, lab_batch = create_batches_rnd(batch_size,data_folder,wav_lst,N_snt,wlen,lab_dict,fact_amp, out_dim)
        yield sig_batch, lab_batch

def create_batches_rnd(batch_size,data_folder,wav_lst,N_snt,wlen,lab_dict,fact_amp, out_dim):
    # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
    sig_batch=np.zeros([batch_size,wlen])
    lab_batch=[]
    snt_id_arr=np.random.randint(N_snt, size=batch_size)
    rand_amp_arr = np.random.uniform(1.0-fact_amp,1+fact_amp,batch_size)
    for i in range(batch_size): 
        # select a random sentence from the list 
        #[fs,signal]=scipy.io.wavfile.read(data_folder+wav_lst[snt_id_arr[i]])
        #signal=signal.astype(float)/32768
        [signal, fs] = sf.read(data_folder+wav_lst[snt_id_arr[i]])
        # accesing to a random chunk
        snt_len=signal.shape[0]
        snt_beg=np.random.randint(snt_len-wlen-1) #randint(0, snt_len-2*wlen-1)
        snt_end=snt_beg+wlen
        sig_batch[i,:]=signal[snt_beg:snt_end]*rand_amp_arr[i]
        y=lab_dict[wav_lst[snt_id_arr[i]]]
        yt = to_categorical(y, num_classes=out_dim)
        lab_batch.append(yt)
    a, b = np.shape(sig_batch)
    sig_batch = sig_batch.reshape((a, b, 1))
    return sig_batch, np.array(lab_batch)

class ValidationCallback(Callback):
    def __init__(self, Batch_dev, data_folder, lab_dict, wav_lst_te, wlen, wshift, class_lay):
        self.wav_lst_te = wav_lst_te
        self.data_folder = data_folder
        self.wlen = wlen
        self.wshift = wshift
        self.lab_dict = lab_dict
        self.Batch_dev = Batch_dev
        self.class_lay = class_lay
    def on_epoch_end(self, epoch, logs={}):
        val = Validation(self.Batch_dev, self.data_folder, self.lab_dict, self.wav_lst_te, self.wlen, self.wshift, self.class_lay, self.model)
        val.validate(epoch)

from conf import *

#np.random.seed(seed)
#from tensorflow import set_random_seed
#set_random_seed(seed)

from keras import models, layers
import numpy as np
import sincnet
from keras.layers import Dense, Dropout, Activation

print('N_filt '+str(cnn_N_filt))
print('N_filt len '+str(cnn_len_filt))
print('FS '+str(fs))
print('WLEN '+str(wlen))

input_shape = (wlen, 1)
out_dim = class_lay[0]
from model import getModel

model = getModel(input_shape, out_dim)
optimizer = RMSprop(lr=lr, rho=0.9, epsilon=1e-8)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics = ['accuracy'])




tb = TensorBoard(log_dir=os.path.join(output_folder,'logs', 'SincNet'))
checkpointer = ModelCheckpoint(
        filepath=os.path.join(output_folder, 'checkpoints',  'SincNet.hdf5'),
        verbose=1,
        save_best_only=False)

validation = ValidationCallback(Batch_dev, data_folder, lab_dict, wav_lst_te, wlen, wshift, class_lay)
callbacks = [tb,checkpointer,validation]


if pt_file!='none':
   model.load_weights(pt_file)

train_generator = batchGenerator(batch_size,data_folder,wav_lst_tr,snt_tr,wlen,lab_dict,0.2, out_dim)
model.fit_generator(train_generator, steps_per_epoch=N_batches, epochs=N_epochs, verbose=1, callbacks=callbacks)