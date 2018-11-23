from tqdm import tqdm
import soundfile as sf
import numpy as np
from conf import *
from model import *

np.random.seed(seed)
class Validation():
    def __init__(self, Batch_dev, data_folder, lab_dict, wav_lst_te, wlen, wshift, class_lay, model, debug=False):
        self.wav_lst_te = wav_lst_te
        self.data_folder = data_folder
        self.wlen = wlen
        self.wshift = wshift
        self.lab_dict = lab_dict
        self.Batch_dev = Batch_dev
        self.class_lay = class_lay
        self.model = model
        self.debug = debug

    def validate(self, epoch=None):
        data_folder = self.data_folder
        wav_lst_te = self.wav_lst_te
        wlen = self.wlen
        wshift = self.wshift
        lab_dict = self.lab_dict
        Batch_dev = self.Batch_dev
        class_lay = self. class_lay
        debug = self.debug
        
        if epoch is None or epoch%N_eval_epoch==0:
            print('Valuating test set...')
            

            snt_te=len(wav_lst_te)

            err_sum = 0
            err_sum_snt = 0
            stn_sum = 0
            if debug:
                print('WLEN: '+str(wlen))
                print('WSHIFT: '+str(wshift))
                pbar = tqdm(total=snt_te)
            for i in range(snt_te):
                [signal, fs] = sf.read(data_folder+wav_lst_te[i])

                signal = np.array(signal)
                lab_batch=lab_dict[wav_lst_te[i]]

                #split signals into chunck
                beg_samp=0
                end_samp=wlen

                N_fr=int((signal.shape[0]-wlen)/(wshift))

                sig_arr=np.zeros([Batch_dev,wlen])

                lab=np.zeros(N_fr+1)+lab_batch
                pout = np.zeros(shape=(N_fr+1,class_lay[-1]))
                count_fr=0
                count_fr_tot=0
                
                
                while end_samp<signal.shape[0]: #for each chunck
                    sig_arr[count_fr,:]=signal[beg_samp:end_samp]
                    beg_samp=beg_samp+wshift
                    end_samp=beg_samp+wlen
                    count_fr=count_fr+1
                    count_fr_tot=count_fr_tot+1
                    if count_fr==Batch_dev: 
                        a,b = np.shape(sig_arr)
                        inp = sig_arr.reshape(a,b,1)
                        inp = np.array(inp)
                        pout[count_fr_tot-Batch_dev:count_fr_tot,:] = self.model.predict(inp, verbose=0)
                        count_fr=0
                        sig_arr=np.zeros([Batch_dev,wlen])

                #Add the last items left 
                if count_fr>0:
                    inp = sig_arr[0:count_fr]
                    a,b = np.shape(inp)
                    inp = inp.reshape(a,b,1)
                    inp = np.array(inp)
                    pout[count_fr_tot-count_fr:count_fr_tot,:] = self.model.predict(inp, verbose=0)

                #Prediction for each chunkc  and calculation of average error
                pred = np.argmax(pout, axis=1)
                err = np.mean(pred!=lab)
                
                #Calculate accuracy on the whole sentence
                best_class = np.argmax(np.sum(pout, axis=0))

                err_sum_snt = err_sum_snt+float((best_class!=lab[0]))
                err_sum = err_sum + err

                stn_sum += 1

                temp_acc_stn = str(round(1-(err_sum_snt/stn_sum), 4))
                temp_acc = str(round(1-(err_sum/stn_sum), 4))
                if debug:
                    pbar.set_description('acc: {}, acc_snt: {}'.format(temp_acc, temp_acc_stn))
                    pbar.update(1)

            #average accuracy
            acc = 1-(err_sum/snt_te)
            acc_snt = 1-(err_sum_snt/snt_te)
            if debug:
                pbar.close()
            if epoch is None:
                print('acc_te: {}, acc_te_snt: {}\n'.format(acc, acc_snt))
            else:
                print('Epoch: {}, acc_te: {}, acc_te_snt: {}\n'.format(epoch, acc, acc_snt))
                with open(output_folder+"/res.res", "a") as res_file:
                    res_file.write("epoch %i, acc_te=%f acc_te_snt=%f\n" % (epoch, acc, acc_snt))   
            return (acc, acc_snt)

def main():
    print("Validation...")
    if pt_file!='none':
        weight_file = pt_file
        input_shape = (wlen, 1)
        out_dim = class_lay[0]
        model = getModel(input_shape, out_dim)
        model.load_weights(weight_file)
        val = Validation(Batch_dev, data_folder, lab_dict, wav_lst_te, wlen, wshift, class_lay, model, True)
        val.validate()
    else:
        print("No PT FILE")
    

if __name__ == "__main__":
    main()