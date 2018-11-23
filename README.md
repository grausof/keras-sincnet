SincNet (M. Ravanelli - Y. Bengio) implementation using Keras Functional Framework v2+
- Models are converted from original caffe networks.
- It supports only Tensorflow backend

# SincNet
SincNet is a neural architecture for processing **raw audio samples**. It is a novel Convolutional Neural Network (CNN) that encourages the first convolutional layer to discover more **meaningful filters**. SincNet is based on parametrized sinc functions, which implement band-pass filters.

<img src="https://github.com/grausof/keras-sincnet/blob/master/SincNet.png" width="400" img align="right">

## References

[1] Mirco Ravanelli, Yoshua Bengio, “Speaker Recognition from raw waveform with SincNet” [Arxiv](http://arxiv.org/abs/1808.00158)

## Cite us
If you use this code or part of it, please cite us!

*Mirco Ravanelli, Yoshua Bengio, “Speaker Recognition from raw waveform with SincNet”* [Arxiv](http://arxiv.org/abs/1808.00158)


## Prerequisites
- Linux
- Python 3.6/2.7
- keras 2.1.6
- Tensorflow 1.10.0
- pysoundfile (``` pip install pysoundfile```)

## How to run a TIMIT experiment
Even though the code can be easily adapted to any speech dataset, in the following part of the documentation we provide an example based on the popular TIMIT dataset.

**1. Run TIMIT data preparation.**

This step is necessary to store a version of TIMIT in which start and end silences are removed and the amplitute of each speech utterance is normalized. To do it, run the following code:

``
python TIMIT_preparation.py $TIMIT_FOLDER $OUTPUT_FOLDER data_lists/TIMIT_all.scp
``

where:
- *$TIMIT_FOLDER* is the folder of the original TIMIT corpus
- *$OUTPUT_FOLDER* is the folder in which the normalized TIMIT will be stored
- *data_lists/TIMIT_all.scp* is the list of the TIMIT files used for training/test the speaker id system.

**2. Run the speaker id experiment.**

- Modify the *[data]* section of *cfg/SincNet_TIMIT.cfg* file according to your paths. In particular, modify the *data_folder* with the *$OUTPUT_FOLDER* specified during the TIMIT preparation. The other parameters of the config file belong to the following sections:
 1. *[windowing]*, that defines how each sentence is splitted into smaller chunks.
 2. *[cnn]*,  that specifies the characteristics of the CNN architecture.
 3. *[dnn]*,  that specifies the characteristics of the fully-connected DNN architecture following the CNN layers.
 4. *[class]*, that specify the softmax classification part.
 5. *[optimization]*, that reports the main hyperparameters used to train the architecture.

- Once setup the cfg file, you can run the speaker id experiments using the following command:

``
python train.py --cfg=cfg/SincNet_TIMIT.cfg
``

The network might take several hours to converge (depending on the speed of your GPU card). 

**3. Results.**

The results are saved into the *output_folder* specified in the cfg file. In this folder, you can find a file (*res.res*) summarizing test accuracy. The model *checkpoints/SincNet.hdf5* is the SincNet model saved after the last iteration. 
Tensorboard can be used to display the loss and accuracy on the train set with the following command:

``
tensorboard --logdir=output_folder/logs
``

Using the cfg file specified above, we obtain the following results:

<img src="https://github.com/grausof/keras-sincnet/blob/master/acc_loss_train.png" width="400" img align="left">

## Where SincNet is implemented?
To take a look into the SincNet implementation you should open the file *sincnet.py* and read the classes *SincConv1D* and the function *sinc*.


## How to use SincNet with a different dataset?
In this repository, we used the TIMIT dataset as a tutorial to show how SincNet works (as in the original code). 
With the current version of the code, you can easily use a different corpus. To do it you should provide in input the corpora-specific input files (in wav format) and your own labels. You should thus modify the paths into the *.scp files you find in the data_lists folder. 

To assign to each sentence the right label, you also have modify the dictionary "*TIMIT_labels.npy*". 
The labels are  specified within a python dictionary that contains sentence ids as keys (e.g., "*si1027*") and speaker_ids as values. Each speaker_id is an integer, ranging from 0 to N_spks-1. In the TIMIT dataset, you can easily retrieve the speaker id from the path (e.g., *train/dr1/fcjf0/si1027.wav* is the sentence_id "*si1027*" uttered by the speaker "*fcjf0*"). For other datasets, you should be able to retrieve in such a way this dictionary containing pairs of speaker and sentence ids.

You should then modify the config file (*cfg/SincNet_TIMIT.cfg*) according to your new paths. Remember also to change the field "*class_lay=462*" according to the number of speakers N_spks you have in your dataset.



## References
[1] SincNet original code written in PyTorch by the autor (https://github.com/mravanelli/SincNet)
[2] Mirco Ravanelli, Yoshua Bengio, “Speaker Recognition from raw waveform with SincNet” [Arxiv](http://arxiv.org/abs/1808.00158)