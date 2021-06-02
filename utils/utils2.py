
import numpy as np
import utils.gen_cutouts as gc

from sklearn import metrics
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stixsans'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
MEAN_TEMP = 2.726 * (10**6)
DEFAULT_FONT = 24

import os
from global_settings import DATA_PATH, FULL_DATA_PATH, FULL_DATA_LABEL_PATH, CNN_MODEL_OUTPUT_DIR, CACHE_FULLDF, CACHE_MAPPED_HALOS, CACHE_FULLDF_DIST2EDGE_CAL

import os
def prepare_data_class(dir_test, num_frequency=3, get_all_components=False, label_fname="1025_hashalo_freq%03i.npy" % 148,
    balanced=False,
    suffix=""):
    """
    read data from dir_test, and prepare data with different noise level (components)
    """
    freqs=[90,148,219]

    def _load_help(name_format):
        paths = [os.path.join(dir_test,  name_format%freq) for freq in freqs]
        ret = [np.load(p) for p in paths]
        #print(paths)
        return ret
    # set file names for data
    #y_data = np.load(dir_test + "1025_hashalo_freq%03i.npy"%148) # y data (labels)
    y_data = np.load(os.path.join(dir_test, label_fname))
    y_data[y_data > 1] = 1
    y_data = y_data.astype(float)
    nsamples = len(y_data)

    #load data into dictionary

    x_data_all = {}
    # load data

    k2uk = 1.0e6
    Tcmb = 2.726

    #load noise (for SPT-3G 1500 sq deg patch, it's [2.8,2.6,6.6]uK-arcmin)
    noises = [np.load(os.path.join(dir_test, "noise_1uK-arcmin{}{}.npy".format(s, suffix))) for s in ["_90","_150", "_220"]]
    noises = [noises[0]*2.8, noises[1]*2.6, noises[2]*6.6]

    #samples has CMB+TSZ
    try:
        com = ['samples','ksz','ir_pts','rad_pts','dust']
        x_data_all['base'] = _load_help("1025_samples_freq%03i{}.npy".format(suffix))

        ksz_comp = _load_help("1025_ksz_freq%03i{}.npy".format(suffix))
        x_data_all['ksz'] = [x_data_all['base'][i] + ksz_comp[i] for i in range(3)]

        ir_comp = _load_help("1025_ir_pts_freq%03i{}.npy".format(suffix))
        x_data_all['ir'] = [x_data_all['ksz'][i] + ir_comp[i] for i in range(3)]

        rad_comp = _load_help("1025_rad_pts_freq%03i{}.npy".format(suffix))
        x_data_all['rad'] = [x_data_all['ir'][i] + rad_comp[i] for i in range(3)]

        dust_comp = _load_help("1025_dust_freq%03i{}.npy".format(suffix))
        x_data_all['dust'] = [x_data_all['rad'][i] + dust_comp[i] for i in range(3)]
    except Exception as err:
        print("error: ", err)
        print("reading only the composite")
        x_data_all['dust'] = _load_help("1025_skymap_freq%03i{}.npy".format(suffix))
    #return x_data_all['dust'], y_data

    x_data = {}
    for com1 in x_data_all.keys():
        # add noise
        x_data[com1] = np.empty((nsamples,num_frequency,10,10),dtype=np.float64)
        if num_frequency == 3:
            for i in range(3):
                x_data[com1][:,i,:,:] = np.squeeze(x_data_all[com1][i]*k2uk*Tcmb) + noises[i]
        else:
            x_data[com1][:,0,:,:] = -np.squeeze(x_data_all[com1][2]*k2uk*Tcmb) - noises[2]
            x_data[com1][:,0,:,:] += np.squeeze(x_data_all[com1][1]*k2uk*Tcmb) + noises[1]
            if num_frequency > 1:
                x_data[com1][:,1,:,:] = -np.squeeze(x_data_all[com1][2]*k2uk*Tcmb) - noises[2]
                x_data[com1][:,1,:,:] += np.squeeze(x_data_all[com1][0]*k2uk*Tcmb) + noises[0]


    if balanced:
        n_pos = int(y_data.sum())
        idx = np.arange(nsamples)
        idx = np.concatenate([idx[y_data==0.0][:n_pos], idx[y_data==1.0]])
        x_data = {k: x_data[k][idx] for k in x_data.keys()}
        return x_data if get_all_components else x_data['dust'], y_data[idx], idx


    return x_data if get_all_components else x_data['dust'], y_data

def prepare_data_class2(dir_test, num_frequency=3, component="skymap", label_fname="1025_hashalo_freq%03i.npy" % 148,
    balanced=False,
    use_noise=True,
    get_test_idx=False,
    suffix=""):
    """
    read data from dir_test, and prepare data with different noise level (components)
    """
    freqs=[90,148,219]

    def _load_help(name_format):
        paths = [os.path.join(dir_test,  name_format%freq) for freq in freqs]
        ret = [np.load(p) for p in paths]
        #print(paths)
        return ret
    # set file names for data
    y_data = np.load(os.path.join(dir_test, label_fname))
    y_data[y_data > 1] = 1
    y_data = y_data.astype(float)
    nsamples = len(y_data)

    #load data into dictionary

    x_data_all = {}
    # load data

    k2uk = 1.0e6
    Tcmb = 2.726

    #load noise (for SPT-3G 1500 sq deg patch, it's [2.8,2.6,6.6]uK-arcmin)
    if use_noise:
        noises = [np.load(os.path.join(dir_test, "noise_1uK-arcmin{}{}.npy".format(s, suffix))) for s in ["_90","_150", "_220"]]
        noises = [noises[0]*2.8, noises[1]*2.6, noises[2]*6.6]
    else:
        noises = [0., 0., 0.]
    #samples has CMB+TSZ
    x_data_all[component] = _load_help("1025_{}_freq%03i{}.npy".format(component, suffix))

    x_data = {}
    for com1 in x_data_all.keys():
        # add noise
        x_data[com1] = np.empty((nsamples,num_frequency,10,10),dtype=np.float64)
        if num_frequency == 3:
            for i in range(3):
                x_data[com1][:,i,:,:] = np.squeeze(x_data_all[com1][i]*k2uk*Tcmb) + noises[i]
        else:
            x_data[com1][:,0,:,:] = -np.squeeze(x_data_all[com1][2]*k2uk*Tcmb) - noises[2]
            x_data[com1][:,0,:,:] += np.squeeze(x_data_all[com1][1]*k2uk*Tcmb) + noises[1]
            if num_frequency > 1:
                x_data[com1][:,1,:,:] = -np.squeeze(x_data_all[com1][2]*k2uk*Tcmb) - noises[2]
                x_data[com1][:,1,:,:] += np.squeeze(x_data_all[com1][0]*k2uk*Tcmb) + noises[0]


    splits = np.asarray([0.8, 0.2])
    splits = np.round(splits / splits.sum() * nsamples).astype(int).cumsum()
    split_idx = np.split(np.arange(nsamples),splits[:-1])
    x_data, x_test = {k: x_data[k][split_idx[0]] for k in x_data.keys()}, {k: x_data[k][split_idx[-1]] for k in x_data.keys()}
    y_data, y_test = y_data[split_idx[0]], y_data[split_idx[-1]]
    nsamples = len(y_data)
    if balanced:
        n_pos = int(y_data.sum())
        idx = np.arange(nsamples)
        idx = np.concatenate([idx[y_data==0.0][:n_pos], idx[y_data==1.0]])
        x_data = {k: x_data[k][idx] for k in x_data.keys()}
        if get_test_idx: return x_data[component], y_data[idx], x_test[component], y_test, idx, split_idx[-1]
        return x_data[component], y_data[idx], x_test[component], y_test, idx
    if get_test_idx:
        return x_data[component], y_data, x_test[component], y_test, split_idx[-1]
    return x_data[component], y_data, x_test[component], y_test



class DataHolder:
    def __init__(self, data, label, idx):
        self.data = data
        self.label = label
        self.idx = idx

    def get(self, which, ratio=None, incl_idx=False):
        curr_idx = self.idx[which]
        y_data = self.label[curr_idx]
        if ratio is not None:
            n_pos = int(y_data.sum())
            idx = np.arange(len(y_data))
            idx = np.concatenate([idx[y_data == 0.0][:int(ratio * n_pos)], idx[y_data == 1.0]])
            curr_idx = curr_idx[idx]
        if incl_idx:
            return self.data[curr_idx], self.label[curr_idx], curr_idx
        return self.data[curr_idx], self.label[curr_idx]


class DataGetter:
    WO_DUST_MAPPING = ("dust", ['samples', 'ksz', 'ir_pts', 'rad_pts'])
    def __init__(self, dir_test, overlap=False):
        self.dir_test = dir_test
        self.overlap = overlap
        self.halocounter = gc.HalosCounter(overlap=overlap)
        df = self.halocounter.get_complete_df()
        if overlap:
            df = df.reset_index().rename(columns={"index": "cutout_id"})
            test_idx = df[(df['cutout_ra'] >= 0.5 * 90) & (df['cutout_dec'] > 0.5 * 90)].index
            train_idx = df[~df.index.isin(test_idx)].index
            n_samples = len(train_idx)
            splits = np.asarray([0.65, 0.1])
            splits = np.round(splits / splits.sum() * n_samples).astype(int).cumsum()
            #print(splits)
            #print(train_idx, len(train_idx))
            split_idx = np.split(train_idx, splits[:-1])
            split_idx = [split_idx[0], split_idx[1], test_idx]
            #print(len(split_idx[0]), len(split_idx[1]), len(split_idx[2]))
            #print(split_idx[0], split_idx[1], split_idx[2])
        else:
            n_samples = df.shape[0]
            splits = np.asarray([0.7, 0.1, 0.2]) # (train ratio, valid ratio, test ratio)
            splits = np.round(splits / splits.sum() * n_samples).astype(int).cumsum()
            split_idx = np.split(np.arange(n_samples), splits[:-1])
            #print(list(map(len, split_idx)), df.shape)
        self.split_idx = {"train":split_idx[0], 'valid':split_idx[1], 'test':split_idx[2]}
        pass

    def get_labels(self, thres=5e13, which='full'):
        if isinstance(thres, float) or isinstance(thres, int):
            thres = ("%0.0e"%(thres)).replace("+", "")
        label_fname = {"5e13": "m5e13_z0.25_y.npy", "2e14":"m2e14_z0.5_y.npy"}[thres]
        y_data = np.load(os.path.join(self.dir_test, label_fname))
        y_data[y_data > 1] = 1
        y_data = y_data.astype(float)
        if which == 'full': return y_data
        return y_data[self.split_idx[which]]

    def get_data(self, component, thres=5e13, use_noise=False, num_frequency=3):

        suffix = "_overlap" if self.overlap else ""
        freqs = [90, 148, 219]

        def _load_help(name_format):
            paths = [os.path.join(self.dir_test, name_format % freq) for freq in freqs]
            return [np.load(p) for p in paths]
        y_data = self.get_labels(thres, which='full')
        nsamples = len(y_data)

        x_data_all = {}
        # load data

        k2uk = 1.0e6
        Tcmb = 2.726

        # load noise (for SPT-3G 1500 sq deg patch, it's [2.8,2.6,6.6]uK-arcmin)
        if use_noise:
            noises = [np.load(os.path.join(self.dir_test, "noise_1uK-arcmin{}{}.npy".format(s, suffix))) for s in
                        ["_90", "_150", "_220"]]
            noises = [noises[0] * 2.8, noises[1] * 2.6, noises[2] * 6.6]
        else:
            noises = [0., 0., 0.]
            # samples has CMB+TSZ
        if isinstance(component, str):
            x_data_all[component] = _load_help("1025_{}_freq%03i{}.npy".format(component, suffix))
        elif isinstance(component,tuple):
            component, lc = component
            x_data_all[component] = _load_help("1025_{}_freq%03i{}.npy".format(lc[0], suffix))
            for cc in lc[1:]:
                tx = _load_help("1025_{}_freq%03i{}.npy".format(cc, suffix))
                assert len(tx) == len(x_data_all[component])
                x_data_all[component] = [x_data_all[component][i] + tx[i] for i in range(len(tx))]

        x_data = {}
        for com1 in x_data_all.keys():
            # add noise
            x_data[com1] = np.empty((nsamples, num_frequency, 10, 10), dtype=np.float64)
            if num_frequency == 3:
                for i in range(3):
                    x_data[com1][:, i, :, :] = np.squeeze(x_data_all[com1][i] * k2uk * Tcmb) + noises[i]
            else:
                x_data[com1][:, 0, :, :] = -np.squeeze(x_data_all[com1][2] * k2uk * Tcmb) - noises[2]
                x_data[com1][:, 0, :, :] += np.squeeze(x_data_all[com1][1] * k2uk * Tcmb) + noises[1]
                if num_frequency > 1:
                    x_data[com1][:, 1, :, :] = -np.squeeze(x_data_all[com1][2] * k2uk * Tcmb) - noises[2]
                    x_data[com1][:, 1, :, :] += np.squeeze(x_data_all[com1][0] * k2uk * Tcmb) + noises[0]
        return DataHolder(x_data[component], y_data, self.split_idx)
    def get_full_df(self):
        df = self.halocounter.get_complete_df().reset_index().rename(columns={"index":"cutout_id"})
        for k, idx in self.split_idx.items():
            df.loc[idx, "which_set"] = k
        return df



class IndexMapper:
    #map cutout_id to the index location
    def __init__(self, overlap=False):
        self.overlap = overlap
        o = DataGetter(overlap)
        self.split_idx = o.split_idx
        self.full_idx = gc.HalosCounter(overlap=overlap).get_complete_df().index
        self.split_idx = {"train":self.full_idx[self.split_idx['train']],
                          "valid":self.full_idx[self.split_idx['valid']],
                          "test":self.full_idx[self.split_idx['test']],
                          "full":self.full_idx}
        self.reverse_idx = {'train':{}, 'valid':{}, 'test':{}, 'full':{}}
        for k in self.split_idx.keys():
            idx = self.split_idx[k]
            for i, d in enumerate(idx):
                self.reverse_idx[k][d] = i

    def get(self, i, which='test'):
        return self.reverse_idx[which][i]


def eval(models, get_test_func, model_weight_paths=None, pred_only=False):
    y_prob_avg = None
    y_probs = []
    x_test, y_test = get_test_func()
    num_nets = len(models)
    for i in range(num_nets):
        model = models[i]
        if model_weight_paths is not None:
            model.load_weights(model_weight_paths[i])
        y_prob = model.predict(x_test)
        y_probs.append(y_prob.squeeze())
        y_prob_avg = y_prob if y_prob_avg is None else y_prob + y_prob_avg
    y_probs = np.stack(y_probs, 0)
    y_prob_avg /= float(num_nets)
    y_pred = (y_prob_avg > 0.5).astype('int32').squeeze() # binary classification
    if pred_only:
        return y_prob_avg
    return summary_results_class(y_probs, y_test), y_pred, y_prob_avg, y_test, x_test, models

def summary_results_class(y_probs, y_test, threshold=0.5, log_roc=False, show_f1=True):
    """
        y_probs: a list of independent predictions
        y_test: true label
        threshold: predict the image to be positive when the prediction > threshold
    """
    # measure confusion matrix
    if show_f1:
        threshold, maxf1 = get_F1(y_probs.mean(0),y_test)
        threshold = threshold - 1e-7

    cm = pd.DataFrame(0, index=['pred0','pred1'], columns=['actual0','actual1'])
    cm_std = pd.DataFrame(0, index=['pred0', 'pred1'], columns=['actual0', 'actual1'])
    #memorizing the number of samples in each case (true positive, false positive, etc.)
    tp_rate, tn_rate = np.zeros(len(y_probs)), np.zeros(len(y_probs))
    for actual_label in range(2):
        for pred_label in range(2):
            cnt = np.zeros(len(y_probs))
            for i in range(len(y_probs)):
                cnt[i] = np.sum(np.logical_and(y_test == actual_label, (y_probs[i] > threshold) == pred_label))
            cm.loc["pred%d"%pred_label,"actual%d"%actual_label] = cnt.mean()
            cm_std.loc["pred%d" % pred_label, "actual%d" % actual_label] = cnt.std()

    print("Confusion matrix (cnts)",cm)
    print("Confusion matrix (stdev of cnts)", cm_std)

    #Measuring the true positive and negative rates,
    #since the false positive/negative rates are always 1 minus these,
    #they are not printed and have the same standard deviation
    for i in range(len(y_probs)):
        pred_i = y_probs[i] > threshold
        tp_rate[i] = np.sum(np.logical_and(y_test==1, pred_i==1)) / np.sum(pred_i==1)
        tn_rate[i] = np.sum(np.logical_and(y_test==0, pred_i==0)) / np.sum(pred_i == 0)
    print("True Positive (rate): {0:0.4f} ({1:0.4f})".format(tp_rate.mean(), tp_rate.std()))
    print("True Negative (rate): {0:0.4f} ({1:0.4f})".format(tn_rate.mean(), tn_rate.std()))



    def vertical_averaging_help(xs, ys, xlen=101):
        """
            Interpolate the ROC curves to the same grid on x-axis
        """
        numnets = len(xs)
        xvals = np.linspace(0,1,xlen)
        yinterp = np.zeros((len(ys),len(xvals)))
        for i in range(numnets):
            yinterp[i,:] = np.interp(xvals, xs[i], ys[i])
        return xvals, yinterp
    fprs, tprs = [], []
    for i in range(len(y_probs)):
        fpr, tpr, _ = metrics.roc_curve(y_test, y_probs[i], pos_label=1)
        fprs.append(fpr)
        tprs.append(tpr)
    new_fprs, new_tprs = vertical_averaging_help(fprs, tprs)

    # measure Area Under Curve (AUC)
    y_prob_mean = y_probs.mean(0)
    auc = metrics.roc_auc_score(y_test, y_prob_mean)
    try:
        auc = metrics.roc_auc_score(y_test, y_prob_mean)
        print()
        print("AUC:", auc)
    except Exception as err:
        print(err)
        auc = np.nan

    #Take the percentiles for of the ROC curves at each point
    new_tpr_mean, new_tpr_5, new_tpr_95 = new_tprs.mean(0), np.percentile(new_tprs, 95, 0), np.percentile(new_tprs, 5, 0)
    # plot ROC curve
    plt.figure(figsize=[12,8])
    lw = 2
    plt.plot(new_fprs, new_tpr_mean, color='darkorange',
                lw=lw, label='ROC curve (area = %0.4f)' % metrics.auc(new_fprs, new_tpr_mean))
    if len(y_probs) > 1:
        plt.plot(new_fprs, new_tpr_95, color='yellow',
                    lw=lw, label='ROC curve 5%s (area = %0.4f)' % ("%", metrics.auc(new_fprs, new_tpr_95)))
        plt.plot(new_fprs, new_tpr_5, color='yellow',
                    lw=lw, label='ROC curve 95%s (area = %0.4f)' % ("%", metrics.auc(new_fprs, new_tpr_5)))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right", fontsize=16)
    plt.grid()
    plt.show()

    #If log flag is set, plot also the log of the ROC curves within some reasonable range
    if log_roc:
        # plot ROC curve
        plt.figure(figsize=[12,8])
        lw = 2
        plt.plot(np.log(new_fprs), np.log(new_tpr_mean), color='darkorange',
                    lw=lw, label='ROC curve (area = %0.4f)' % metrics.auc(new_fprs, new_tpr_mean))
        if len(y_probs) > 1:
            plt.plot(np.log(new_fprs), np.log(new_tpr_95), color='yellow',
                        lw=lw, label='ROC curve 5%s (area = %0.4f)' % ("%", metrics.auc(new_fprs, new_tpr_95)))
            plt.plot(np.log(new_fprs), np.log(new_tpr_5), color='yellow',
                        lw=lw, label='ROC curve 95%s (area = %0.4f)' % ("%", metrics.auc(new_fprs, new_tpr_5)))
        #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([-5, -3])
        plt.ylim([-1, 0.2])
        plt.xlabel('Log False Positive Rate', fontsize=16)
        plt.ylabel('Log True Positive Rate', fontsize=16)
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right", fontsize=16)
        plt.grid()
        plt.show()
    return (auc,maxf1) if show_f1 else auc, (tp_rate.mean(),tn_rate.mean()), new_fprs, new_tprs



#=======================================================Prediction criteria here
import numpy as np
import pickle
import pandas as pd
from scipy.optimize import minimize


def _get_Fbeta(y, yhat, beta=1., debug=False, get_raw=False):
    TP = ((y == 1) & (yhat == 1)).sum()
    FP = ((y == 0) & (yhat == 1)).sum()
    TN = ((y == 0) & (yhat == 0)).sum()
    FN = ((y == 1) & (yhat == 0)).sum()
    if debug: print("TP: {}, FP:{}, TN:{}, FN:{}".format(TP, FP, TN, FN))
    if FP+TP == 0 or TP + FN==0 or TP == 0: return -1.
    precision = (TP) / (FP + TP).astype(float)
    recall = (TP) / (TP + FN).astype(float)
    if debug:
        print("TP={}; FP={}; TN={}; FN={}; precision={};recall={}".format(((y == 1) & (yhat == 1)).sum(),
                                                                          ((y == 0) & (yhat == 1)).sum(),
                                                                          ((y == 0) & (yhat == 0)).sum(),
                                                                          ((y == 1) & (yhat == 0)).sum(), precision,
                                                                          recall))
    if get_raw: return precision, recall, (1 + beta ** 2) * (precision * recall) / (beta * precision + recall)
    return (1 + beta ** 2) * (precision * recall) / (beta * precision + recall)


def get_F1(y_pred, y, xlim=None, method='cnn', mass_thresh='5e13', plot=True,
           save_path=None, xlabel=None, get_raw=False, font=DEFAULT_FONT):
    plt.rcParams.update({'font.size': font})
    if xlim is None:
        xlim = (0, 0.997)
        x = np.linspace(xlim[0], xlim[1])
    elif isinstance(xlim, tuple):
        x = np.linspace(xlim[0], xlim[1])
    else:
        x = xlim
    Fscore = lambda x: _get_Fbeta(y, (y_pred > x).astype(int))

    y = np.asarray([Fscore(xx) for xx in x]).clip(0.)
    if plot:
        f = plt.figure(figsize=(8, 5))

        plt.plot(x, y)
        plt.xlim(x[0], x[-1])
        if xlabel:
            plt.xlabel(xlabel)
        else:
            plt.xlabel('%s Threshold' % ("CNN Prob" if method == 'cnn' else "MF S/N"))
        plt.ylabel('F1 Score', fontsize=font)
        if save_path is not None: plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.show(block=True)
    if get_raw: return x, y
    return x[np.argmax(y)], np.max(y)

def stack_F1(xmf, ymf, xcnn, ycnn, save_path=None, font=DEFAULT_FONT, title="", hist={}, nxbins=20):
    lns = []
    fig = plt.figure(figsize=(8,5))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    plt.rcParams.update({'font.size': font})

    lns.extend(ax1.plot(xmf, ymf, label='MF', color='purple'))
    ax1.set_xlabel("MF S/N Ratio")
    ax1.set_ylabel("F1 Score")
    lns.extend(ax2.plot(xcnn, ycnn, label='CNN', color='black'))
    ax2.set_xlabel("CNN Prob")


    import matplotlib.patches as mpatches
    if 'MF' in hist:
        assert 'CNN' in hist
        ax12 = ax1.twinx()
        bins = np.linspace(*ax1.get_xlim(), num=nxbins)
        #ax12.set_ylim(0, 100000)
        ax12.hist(hist['MF'][~pd.isnull(hist['MF'])], alpha=0.3, bins=bins, color='purple',
                  #weights=np.ones(len(hist['MF']))/len(hist['MF'])
                  )
        lns.append(mpatches.Patch(color='purple', label='MF Score Dist.', alpha=0.3))
        ax12.set_yscale('log')


        ax22 = ax2.twinx()
        bins = np.linspace(*ax2.get_xlim(), num=nxbins)
        #ax22.set_ylim(0, 100000)
        ax22.hist(hist['CNN'], alpha=0.3, bins=bins, color='black',
                  #weights=np.ones(len(hist['CNN']))/len(hist['CNN'])
                  )
        lns.append(mpatches.Patch(color='black', label='CNN Score Dist.', alpha=0.3))

        ax22.set_yscale('log')
        if ax12.get_ylim()[1] > ax22.get_ylim()[1]:
            ax22.set_ylim(ax12.get_ylim())
        else:
            ax12.set_ylim(ax22.get_ylim())



        #ylim1 = ax12.get_ylim()
        #ylim2 = ax22.get_ylim()
        #ylim = (min(ylim1[0], ylim2[0]), max(ylim1[1], ylim2[1]))
        #print(ylim)
        #for _temp in [ax12, ax22]:
            #_temp.set_ylim(ylim)
            #_temp.set_yscale('log')
        ax12.set_ylabel("Counts", fontsize=font)
        #ax12.set_yscale('log')

    labs = [l.get_label() for l in lns]
    plt.title(title)
    plt.legend(lns, labs, loc='lower center', prop={"size":font-8})
    if save_path is not None: plt.savefig(save_path, dpi=500, bbox_inches="tight")
    return

def get_F1_CNN_and_MF(vdf, col_cnn='score_wdust (trained>%s)', col_mf ='mf_peaksig', col_label='Truth(>%s)',
                      mass_thresh='5e13', method='and', save_path=None, font=DEFAULT_FONT):
    plt.rcParams.update({'font.size': font})
    import itertools
    if mass_thresh == '5e13':
        cnn_range = (0, 0.997)
        mf_range = (3, 15)
    else:
        #cnn_range = (0.4, 0.8)
        #mf_range = (3, 15)
        cnn_range = (0.2, 0.9)
        mf_range = (3, 25)
    cnn_range = np.linspace(cnn_range[0], cnn_range[1])
    mf_range = np.linspace(mf_range[0], mf_range[1])
    #criteria = itertools.product(cnn_range, mf_range)
    criteria = [(c,m) for c in cnn_range for m in mf_range]
    if method == 'or':
        Fscore = lambda cc, mc: _get_Fbeta(vdf[col_label], ((vdf[col_cnn] > cc) | (vdf[col_mf] > mc)).astype(int))
    elif method == 'and':
        Fscore = lambda cc, mc: _get_Fbeta(vdf[col_label], ((vdf[col_cnn] > cc) & (vdf[col_mf] > mc)).astype(int))
    elif method == 'rankproduct':
        rnk_cnn = vdf[col_cnn].rank() / len(vdf)
        rnk_mf = vdf[col_mf].rank() / float(vdf[col_mf].count())
        return get_F1(rnk_cnn * rnk_mf, vdf[col_label], xlim=(0.7, .985), xlabel='rank product', save_path=save_path, font=font)
    cnn_x = np.asarray([c[0] for c in criteria])
    mf_y = np.asarray([c[1] for c in criteria])
    vals = np.asarray([Fscore(cc,mc) for cc,mc in criteria])

    cm = plt.cm.get_cmap('YlGn')
    sc = plt.scatter(cnn_x, mf_y, c=vals, cmap=cm)
    plt.scatter(*criteria[np.argmax(vals)], s=100, c='black', marker='x', linewidth=3)
    cbar = plt.colorbar(sc)
    cbar.set_label("F1 Score", rotation=270, labelpad=20)
    plt.xlabel("CNN Threshold")
    plt.ylabel("MF Threshold")
    if save_path is not None: plt.savefig(save_path, dpi=500, bbox_inches="tight")
    return criteria[np.argmax(vals)], np.max(vals)


import glob
import ipdb


class PytorchResultReader(object):
    def __init__(self, data_dir = CNN_MODEL_OUTPUT_DIR,
                 exp_name="ratio1-20_convbody=R-50-C4_SGD_lr=0.005_wd=0.003_steps=1000-4000_comp=skymap",
                 xlim=(0.5, 0.7)):
        self.data_dir = data_dir
        if exp_name is None:
            exp_name = [f for f in os.listdir(self.data_dir) if "_lr" in f and '_wd' in f]
        self.exp_names = [exp_name] if isinstance(exp_name,str) else exp_name
        labels = pd.read_pickle(FULL_DATA_LABEL_PATH)
        test_labels, val_labels = labels[labels['which'] == 'test'], labels[labels['which'] == 'valid']
        np.random.seed(0)
        self.labels = test_labels.iloc[np.random.permutation(len(test_labels))]
        np.random.seed(0)
        self.val_labels = val_labels.iloc[np.random.permutation(len(val_labels))]
        self.xlim = xlim

    def _read_one(self, exp_name, xlim=None):
        dir_path = os.path.join(self.data_dir, exp_name)
        fs = sorted(glob.glob(os.path.join(dir_path, 'results', 'epoch*.pkl')),
                    key=lambda x: int(x.replace(".pkl", "").split("epoch")[1]))
        if len(fs) ==0: return None, None, None
        dfs = {w: pd.DataFrame(columns=['acc', 'loss', 'F1', "F1_thres"], dtype=float) for w in ['test', 'val_ret']}
        preds, val_preds = {}, {}
        iter_ser = {}
        for f in fs:
            epoch = int(os.path.basename(f).replace(".pkl", "").split("epoch")[1])
            res = pd.read_pickle(f)
            for which in dfs.keys():
                for key in ['acc', 'loss']:
                    dfs[which].loc[epoch, key] = res[which][key]
                y_pred, y = res[which]['y_pred'], res[which]['y']
                dfs[which].loc[epoch, 'F1_thres'], dfs[which].loc[epoch, 'F1'] = get_F1(y_pred, y, plot=False, xlim=xlim)
            if f == fs[-1]:
                for which in ['train', 'val']: dfs[which] = pd.DataFrame(res[which]).astype(float)
            if 'true' not in preds:
                preds['true'] = res['test']['y']
                val_preds['true'] = res['val_ret']['y']
            preds[epoch] = res['test']['y_pred']
            val_preds[epoch] = res['val_ret']['y_pred']
            iter_ser[epoch] = pd.DataFrame(res['train']).index.max()

        min_len = min(len(res['test']['y']), len(self.labels))
        assert min_len == (res['test']['y'][:min_len] == self.labels['y'].values[:min_len]).sum()
        preds = pd.DataFrame(preds).iloc[:min_len]
        preds.index = self.labels.index[:min_len]
        preds['which'] = 'test'

        val_min_len = min(len(res['val_ret']['y']), len(self.val_labels))
        assert val_min_len == (res['val_ret']['y'][:val_min_len] == self.val_labels['y'].values[:val_min_len]).sum()
        val_preds = pd.DataFrame(val_preds).iloc[:val_min_len]
        val_preds.index = self.val_labels.index[:val_min_len]
        val_preds['which'] = 'valid'

        preds = pd.concat([preds, val_preds])
        preds.index.name = 'cutout_id'
        return dfs, preds, iter_ser

    def get_all(self):
        results = {exp: self._read_one(exp, self.xlim) for exp in self.exp_names}
        return results

    def get_best(self):
        best_test_results, best_test_F1 = {}, -1.
        best_val_results, best_val_F1 = {}, -1.
        for exp in self.exp_names:
            res = self._read_one(exp, self.xlim)
            if res[0] is None: continue
            best_epoch = res[0]['test'].sort_values('F1', ascending=False).index[0]
            if res[0]['test'].loc[best_epoch, 'F1'] > best_test_F1:
                best_test_F1 = res[0]['test'].loc[best_epoch, 'F1']
                best_test_results['pred'] = res[1].reindex(columns=['true', best_epoch, 'which'])
                best_test_results['stat'] = res[0]['test'].loc[best_epoch]
                best_test_results['name'] = exp

            best_epoch = res[0]['val_ret'].sort_values('F1', ascending=False).index[0]
            if res[0]['val_ret'].loc[best_epoch, 'F1'] > best_val_F1:
                best_val_F1 = res[0]['val_ret'].loc[best_epoch, 'F1']
                best_val_results['pred'] = res[1].reindex(columns=['true', best_epoch, 'which'])
                best_val_results['stat'] = res[0]['val_ret'].loc[best_epoch]
                best_val_results['name'] = exp
        return best_val_results, best_test_results


import time, sys
class ProgressBar:
    def __init__(self, iterable, taskname=None, barLength=40, stride = 50):
        self.l = iterable
        try:
            self.n = len(self.l)
        except TypeError:
            self.l = list(self.l)
            self.n = len(self.l)
        self.cur = 0
        self.starttime = time.time()
        self.barLength = barLength
        self.taskname = taskname
        self.last_print_time = time.time()
        self.stride = stride

    def __iter__(self):
        return self
    def _update_progress(self):
        status = "Done...\r\n" if self.cur == self.n else "\r"
        progress = float(self.cur) / self.n
        curr_time = time.time()

        block = int(round(self.barLength * progress))
        text = "{}Percent: [{}] {:.2%} Used Time:{:.2f} seconds {}".format("" if self.taskname is None else "Working on {}. ".format(self.taskname),
                                                                      "#" * block + "-"*(self.barLength - block),
                                                                      progress, curr_time - self.starttime, status)
        sys.stdout.write(text)
        sys.stdout.flush()

    def __next__(self):
        if self.cur % self.stride == 0:
            self._update_progress()
        if self.cur >= self.n:
            raise StopIteration
        else:
            self.cur += 1
            return self.l[self.cur - 1]

def match(mf_df, dist=1, data_dir="../data/", cache_dir=None):
    """
    Clean up the matches such that we assign the largest (highest mass) halo to the MF signal

    :param mf_df:
        MF

    :param dist:
    :param data_dir:

    :param cache_dir:

    :return:
    """
    from astropy.coordinates import SkyCoord
    from astropy import units as units
    if cache_dir is None: cache_dir = os.path.join(data_dir, 'cache')
    if not os.path.isdir(cache_dir): os.makedirs(cache_dir)

    run_min_sig = 3 if np.min(mf_df['mf_peaksig']) < 3.5 else 4
    MF_match_cache_path = os.path.join(cache_dir, "MF_match_{}sig_{}arcmin.pkl".format(run_min_sig, dist))

    if not os.path.isfile(MF_match_cache_path):
        halo_data = pd.read_csv(data_dir + 'halo_sz.ascii', sep='\s+', header=None,
                                names=['redshift', 'ra', 'dec', 'Mvir', 'M500'], usecols=[0, 1, 2, 10, 32])
        halo_coord = SkyCoord(ra=halo_data['ra'].values * units.degree, dec=halo_data['dec'].values * units.degree)
        mf_coord = SkyCoord(ra=mf_df['mf_ra'].values * units.degree, dec=mf_df['mf_dec'].values * units.degree)

        #idx, sep_2d, dist_3d = mf_coord.match_to_catalog_sky(halo_coord)
        idxmf, idxhalo, sep_2d, dist_3d = halo_coord.search_around_sky(mf_coord, dist * units.arcmin)
        bad_idxmf = mf_df.index.difference(idxmf)
        mf_df.shape, len(np.unique(idxmf)) + len(bad_idxmf)

        mf_df = mf_df.reindex(columns=['mf_peaksig', 'mf_dec', 'mf_ra'])
        print(mf_df.dropna().shape)

        n_mf = len(mf_df)
        matched_halos = set()
        match_list = []
        for ii in ProgressBar(range(n_mf - 1, -1, -1)):
            idxhalo_this = idxhalo[idxmf == ii]
            halos_match = halo_data.iloc[idxhalo_this].copy()
            while not halos_match.empty:
                idx_mostmass = halos_match['Mvir'].idxmax()
                if idx_mostmass in matched_halos and len(halos_match) > 1:
                    halos_match.drop(idx_mostmass, inplace=True)
                    continue
                matched_halos.add(idx_mostmass)
                match_list.append(
                    np.concatenate(([idx_mostmass], halos_match.loc[idx_mostmass].values, [ii], mf_df.loc[ii].values)))
                break
        mfhalo_df = pd.DataFrame(match_list,
                                 columns=['halo_id', 'redshift', 'ra_halo', 'dec_halo', 'Mvir', 'M500',
                                          'mf_id', 'StoN', 'mf_dec', 'mf_ra'])
        mfhalo_df = pd.concat([mfhalo_df, mf_df.reindex(bad_idxmf).rename(columns={"mf_peaksig": "StoN"})])
        mfhalo_df.to_pickle(MF_match_cache_path)
    mfhalo_df = pd.read_pickle(MF_match_cache_path)
    return mfhalo_df


def show_cutouts(df, get_x_func, n=5, cutout_size=8./60, show_freq=True, save_path=None, font=DEFAULT_FONT):

    f = plt.figure(figsize=(15, 5 * n))
    for i in range(n):
        r = df.iloc[i]
        print("Cutout {} has features: Mvir={:.1e} redshift={} rvir={}".format(r['cutout_id'], r['Mvir'], r['redshift'],
                                                                               r['rvir']))
        img = get_x_func(r['cutout_id'])
        for c in range(3):
            f.add_subplot(n, 3, 3 * i + c + 1)
            #plt.imshow(img[:,:,c], cmap='gray')
            im = plt.imshow(img[:, :, c], cmap='gray', extent=[cutout_size / 2, -cutout_size / 2, cutout_size / 2, -cutout_size / 2])
            plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            plt.scatter([r['CL_ra'] - r['cutout_ra']], [r['CL_dec'] - r['cutout_dec']], s=200, c='red', marker='x', linewidth='3')
            plt.title("freq %d GHz" % ({0:90,1:148,2:219}[c]), fontsize=font)
            if c == 0: plt.ylabel("img %d" % r['cutout_id'], fontsize=font)

        #plt.scatter([r['CL_ra'] - r['cutout_ra']], [r['CL_dec'] - r['cutout_dec']])
        #plt.xlim(cutout_size / 2, -cutout_size / 2)
        #plt.ylim(cutout_size / 2, -cutout_size / 2)
        #plt.xlabel("ra")
        #plt.ylabel("dec")
        #plt.title("position in the cutout", fontsize=16)
    if save_path is not None: plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.show(block=True)

def _get_cax_for_colobar(ax, fig):
    return fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])


from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid
import ipdb
def show_cutout_full(r, get_comp_func, cutout_size=8./60, save_path=None, font=DEFAULT_FONT, separate_cbar=None, mark=True,
                     normalization=None,
                     components = ['samples', 'ksz', 'ir_pts', 'rad_pts', 'dust'],
                     override_name_map={},
                     adjust=None, width_multiple=1.1):
    assert normalization is None or normalization in {"log"}

    plt.rcParams.update({'font.size': font})
    #get_x_func = lambda c: ..
    #components = ['samples', 'ksz', 'ir_pts', 'rad_pts', 'dust']#, 'skymap']
    name_map = {"samples":"CMB + tSZ", "ksz":"+kSZ", "ir_pts":"+Infrared Galaxies", "rad_pts":"+Radio Galaxies", "dust":"+Galactic Dust"}
    name_map.update(override_name_map)
    nc = len(components)

    fig = plt.figure(figsize=(4 * nc * width_multiple, 4 * 3))

    #fig, axes = plt.subplots(nrows=3, ncols=nc, figsize=(4 * nc, 4 * 3))

    comps = {c: get_comp_func(c) for c in components}
    if separate_cbar is None:
        kwargs = [{"vmin": min([comps[c].min() for c in components]),
                  "vmax": max([comps[c].max() for c in components]) }for _ in range(3)]
        raise NotImplementedError
        grid_kwargs = {}
    elif separate_cbar == 'row':
        kwargs = [{"vmin": min([comps[c][:,:,freq].min() for c in components]),
                   "vmax": max([comps[c][:,:,freq].max() for c in components])} for freq in range(3)]
        grid_kwargs = {"cbar_mode":"edge", "direction":"row"}
        default_adjust = lambda f: None

        grid = ImageGrid(fig, 111,
                         nrows_ncols=(3, nc),
                         axes_pad=0.2,
                         share_all=True,
                         cbar_location="right",
                         cbar_size="4%",
                         cbar_pad=0.1,
                         **grid_kwargs)  # https://stackoverflow.com/questions/45396103/imagegrid-with-colorbars-only-on-some-subplots
        get_ax = lambda ic, ir: grid[nc * ir + ic]
    elif separate_cbar == 'col':
        raise NotImplementedError
    else:
        assert separate_cbar == 'each'
        kwargs = [{},{},{}]
        #grid_kwargs = {"cbar_mode": "each"}
        default_adjust = lambda f: f.subplots_adjust(hspace=-0.1, wspace=0.25)
        get_ax = lambda ic, ir: fig.add_subplot(3, nc, nc * ir + ic + 1)
    if adjust is None: adjust = default_adjust



    if normalization == 'log':
        assert separate_cbar == 'row'
        offset = {}
        for i, kk in enumerate(kwargs):
            vmin, vmax = kk['vmin'], kk['vmax']
            offset[i] = vmin - 1
            kk['vmin'] -= offset[i]
            kk['vmax'] -= offset[i]


        def transform(img, freq):
            return img[:, :, freq] - offset[freq]

        def set_cbar_ticks(cbar, freq):
            vmin, vmax = kwargs[freq]['vmin'], kwargs[freq]['vmax']
            bias = offset[freq]
            ticks = np.logspace(np.log10(vmin), np.log10(vmax), 5)
            ticklabels = ["%.1f"%(t + bias) for t in ticks]
            cbar.ax.set_yscale('log')
            cbar.ax.set_yticks(ticks)
            cbar.ax.set_yticklabels(ticklabels)

    else:
        def transform(img, freq):
            return img[:, :, freq]

    import matplotlib.colors as colors
    import matplotlib.ticker as ticker

    for i, c in enumerate(components):
        img = get_comp_func(c)
        for freq in range(3):
            cplt = get_ax(i, freq)
            norm = colors.LogNorm(clip=True, **kwargs[freq]) if normalization == 'log' else None
            im = cplt.imshow(transform(img, freq), cmap='gray', norm=norm,
                             extent=[cutout_size / 2, -cutout_size / 2, cutout_size / 2, -cutout_size / 2],
                             **kwargs[freq])

            cplt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)


            if mark: cplt.scatter([r['CL_ra'] - r['cutout_ra']], [r['CL_dec'] - r['cutout_dec']], s=200, c='red', marker='x', linewidth='3')
            if separate_cbar == 'each':
                divider = make_axes_locatable(cplt)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            elif c == components[-1] and separate_cbar == 'row':
                if normalization == 'log':
                    cbar = cplt.cax.colorbar(im, norm=norm, format=ticker.LogFormatterMathtext())
                    set_cbar_ticks(cbar, freq)
                else:
                    cbar = cplt.cax.colorbar(im, norm=norm)
            else:
                cbar = None
            if cbar is not None:
                cbar.ax.tick_params(labelsize=font-5)
                if freq == 0: cbar.ax.set_title('$\mu$K', fontsize=font - 4)

            if freq == 0: cplt.set_title(name_map[c])  # , fontsize=font)
            if i == 0: cplt.set_ylabel("%d GHz"%{0:90,1:148,2:219}[freq])#, fontsize=font)

            continue
    if not separate_cbar:
        # put colorbar at desire position
        divider = make_axes_locatable(cplt)
        fig.subplots_adjust(right=0.95)
        cbar_ax = fig.add_axes([0.96, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        #cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('$\mu$K', rotation=270, labelpad=15)#, fontsize=font)
    if save_path is not None: plt.savefig(save_path, dpi=500,bbox_inches="tight")
    #fig.tight_layout()
    #fig.subplots_adjust(top=1.00, bottom=0.)
    #fig.subplots_adjust(hspace=-0.1, wspace=0.25)
    adjust(fig)
    plt.show()


def show_range(df, get_x_func, n=5, cnn_prob=(0.6, 1.0), mf_sn=(3., 5.), which='test', tp=True,
               CNN_col='pred', MF_col='StoN', label_col='y', cutout_size=8./60, save_path=None):
    ss = "Sampling cutouts such that"
    if cnn_prob is not None:
        ss = ss + " CNN prob in ({},{}),".format(cnn_prob[0], cnn_prob[1])
        df = df[(df[CNN_col] > cnn_prob[0]) & (df[CNN_col] < cnn_prob[1])]
    if mf_sn is not None:
        ss = ss + " MF S/N in ({}, {}),".format(mf_sn[0], mf_sn[1])
        df = df[(df[MF_col] > mf_sn[0]) & (df[MF_col] < mf_sn[1])]
    if tp is not None:
        ss = ss + (" has halo" if tp else " does not have halo")
        df = df[df[label_col]] if tp else df[~df[label_col]]
    print(ss)
    print("there are {} such cutouts ".format(len(df)))
    show_cutouts(df, get_x_func, n,cutout_size, save_path=save_path)
    return df

def false_color(x, params=None):
    for i in range(3):
        try:
            _min = params[i]['min']
            _max = params[i]['max']
        except:
            _min = np.min(x[:,:,i])
            _max = np.max(x[:,:,i])
        x[:, :, i] = x[:, :, i].clip(_min, _max)
        x[:,:,i] = (x[:,:,i] - _min)/(_max - _min) * 255
    return np.round(x).astype(int)#[:,:,0]


def false_color_log(x, params=None):
    for i in range(3):
        try:
            _min = params[i]['min']
            _max = params[i]['max']
        except:
            _min = np.min(x[:,:,i])
            _max = np.max(x[:,:,i])
        x[:, :, i] = np.log(x[:,:,i] - _min + 1.)
        x[:, :, i] = (x[:, :, i] - np.min(x[:, :, i])) / (np.max(x[:, :, i]) - np.min(x[:, :, i])) * 255
        #x[:,:,i] = (x[:,:,i] - _min)/(_max - _min) * 255
    return np.round(x).astype(int)#[:,:,0]

def single_channel_log(x, params=None):
    try:
        _min = params['min']
        _max = params['max']
    except:
        _min = np.min(x)
        _max = np.max(x)
    x = np.log(x - _min + 1.)
    return x

def make_false_color_params1(cutouts):
    params = [{} for i in range(3)]
    for k in cutouts.keys():
        for i in range(3):
            try:
                params[i]['max'] = max(params[i]['max'], np.max(cutouts[k][:,:,i]))
            except:
                params[i]['max'] = np.max(cutouts[k][:, :, i])
            try:
                params[i]['min'] = min(params[i]['min'], np.min(cutouts[k][:,:,i]))
            except:
                params[i]['min'] = np.min(cutouts[k][:, :, i])
    print(params)
    return params


def make_false_color_params(cutouts):
    channels = [[], [], []]
    for k in cutouts.keys():
        for i in range(3):
            channels[i].append(cutouts[k][:,:,i].flatten())
    channels = [np.concatenate(x) for x in channels]
    params = [{"max": np.percentile(channels[i], 99.5),
               "min": np.percentile(channels[i], 0.5)} for i in range(3)]
    print("max pixel val = %f, min=%f; 148 max=%f, min=%f; 219 max=%f, min=%f"%(params[0]['max'], params[0]['min'], params[1]['max'], params[1]['min'],params[2]['max'], params[2]['min']))
    return params


def make_false_color_params_bad(cutouts):
    channels = [[], [], []]
    for k in cutouts.keys():
        for i in range(3):
            channels[i].append(cutouts[k][:,:,i].flatten())
    channels = [np.concatenate(x) for x in channels]
    params = [{"max": np.percentile(channels[i], 99.5),
               "min": np.percentile(channels[i], 0.5)} for i in range(3)]
    print("max pixel val = %f, min=%f; 148 max=%f, min=%f; 219 max=%f, min=%f"%(params[0]['max'], params[0]['min'], params[1]['max'], params[1]['min'],params[2]['max'], params[2]['min']))
    return [{"max": max([params[i]['max'] for i in range(3)]), "min": min([params[i]['min'] for i in range(3)])}] * 3
def show_range_by_feature(df, CNN_pred, MF_pred, get_x_func, n=5, feature='redshift',
                          ycol ='y', save_path=None):
    import matplotlib.pyplot as plt
    if isinstance(CNN_pred,str): CNN_pred = df[CNN_pred]
    if isinstance(MF_pred, str): MF_pred = df[MF_pred]
    percs = np.linspace(0., 1., n+1)[:n] + 0.5 / n
    cols = ["MF_TP", "MF_FN", "CNN_TP", "CNN_FN"]
    col2idxs = {"MF_TP": df[df[ycol] & MF_pred].index, "MF_FN": df[df[ycol] & ~MF_pred].index,
                "CNN_TP": df[df[ycol] & CNN_pred].index, "CNN_FN": df[df[ycol] & ~CNN_pred].index}
    rows = ["%.2f%%-tile"%(p * 100) for p in percs]
    import ipdb
    #ipdb.set_trace()
    fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), figsize=(3 * len(cols), 3 * len(rows)))

    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row, rotation=90, size='large')

    for row_i, row in enumerate(rows):
        for col_i, col in enumerate(cols):
            #print(row_i, col_i)
            tdf =df.reindex(col2idxs[col])
            curr_percentile_val = tdf[feature].quantile(percs[row_i])
            r = tdf.loc[tdf[feature].map(lambda x: (x - curr_percentile_val) ** 2).idxmin()]
            print("Cutout {} has features: Mvir={:.1e} redshift={} rvir={}".format(r['cutout_id'], r['Mvir'], r['redshift'],
                                                                                   r['rvir']))
            axes[row_i, col_i].imshow(false_color(get_x_func(r['cutout_id'])), cmap='gray')
            axes[row_i, col_i].set_xlabel("%s=%f"%(feature, r[feature]))

    fig.tight_layout()
    if save_path is not None: plt.savefig(save_path, dpi=500,bbox_inches="tight")
    plt.show()

def show_range_by_feature2(df, pred_cols, get_x_func, n=5, feature='redshift', use_log=False,
                              ycol='y', save_path=None):
        import matplotlib.pyplot as plt
        print("There are %d such cutouts in total"%(len(df)))
        print(
            "Below are examples whose %s are at different percentiles (shown on the left side of the left-most column) within these %d cutouts." % (
            feature, len(df)))
        print("%s increases from top to bottom" % feature)

        percs = np.linspace(0., 1., n + 1)[:n] + 0.5 / n
        cols , col2idxs = [], {}
        for k in pred_cols.keys():
            if isinstance(pred_cols[k], str): pred_cols[k] = df[pred_cols[k]]
            cols.extend(["%s_TP"%k, "%s_FN"%k])
            col2idxs['%s_TP'%k] = df[df[ycol] & (pred_cols[k])].index
            col2idxs['%s_FN' % k] = df[df[ycol] & (~pred_cols[k])].index
            if feature not in {"redshift", "tSZ", "Mvir", "rvir"}:
                cols.extend(["%s_FP" % k, "%s_TN" % k])
                col2idxs['%s_FP' % k] = df[(~df[ycol]) & pred_cols[k]].index
                col2idxs['%s_TN' % k] = df[(~df[ycol]) & ~pred_cols[k]].index
            #else:
                #cols.extend(["%s_FP" % k, "%s_TN" % k])
        old_cols = cols.copy()
        cols = []
        for k in old_cols:
            if len(col2idxs[k]) > 0:
                cols.append(k)
            else:
                col2idxs.pop(k)
        rows = ["%.2f%%-tile" % (p * 100) for p in percs]
        import ipdb
        # ipdb.set_trace()
        fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), figsize=(3 * len(cols), 3 * len(rows)))
        print("from left to right we have in each column {}".format(", ".join(cols)))
        for ax, col in zip(axes[0], cols):
            ax.set_title(col)

        for ax, row in zip(axes[:, 0], rows):
            ax.set_ylabel(row, rotation=90, size='large')
        cutout_size = 8. / 60
        cutouts, rs = {}, {}
        for row_i, row in enumerate(rows):
            for col_i, col in enumerate(cols):
                # print(row_i, col_i)
                tdf = df.reindex(col2idxs[col])
                curr_percentile_val = tdf[feature].quantile(percs[row_i])
                r = tdf.loc[tdf[feature].map(lambda x: (x - curr_percentile_val) ** 2).idxmin()]
                #print("Cutout {} has features: Mvir={:.1e} redshift={} rvir={}".format(r['cutout_id'], r['Mvir'], r['redshift'], r['rvir']))
                cutouts[(row_i, col_i)], rs[(row_i, col_i)] = get_x_func(r['cutout_id']), r
        if use_log:
            fc = lambda x: false_color_log(x)
        else:
            _fc_params = make_false_color_params(cutouts)
            fc = lambda x: false_color(x, _fc_params)
        for row_i, row in enumerate(rows):
            for col_i, col in enumerate(cols):
                # print(row_i, col_i)
                r = rs[(row_i, col_i)]
                axes[row_i, col_i].imshow(fc(cutouts[(row_i, col_i)]), extent=[cutout_size/2, -cutout_size/2, cutout_size/2, -cutout_size/2])
                axes[row_i, col_i].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
                if not pd.isnull(r['CL_ra']):
                    axes[row_i, col_i].scatter([r['CL_ra'] - r['cutout_ra']], [r['CL_dec'] - r['cutout_dec']], s=200, edgecolors='red', marker='o', facecolors='none')
                if feature in ['rvir' ,'redshift', 'Mvir', 'tSZ']:
                    axes[row_i, col_i].set_xlabel("%s=%.2E" % (feature, r[feature]))
                else:
                    axes[row_i, col_i].set_xlabel("%s=%f" % (feature, r[feature]) + ", Mass=%.2E" % ("Mvir", r["Mvir"]))

        fig.tight_layout()
        if save_path is not None: plt.savefig(save_path, dpi=500, bbox_inches="tight")
        plt.show()

def show_false_color_2x2(rows, get_x_func, save_path=None, font=DEFAULT_FONT):
    plt.rcParams.update({'font.size': font})
    cutout_size = 8. / 60
    cutouts = {k: get_x_func(rows[k].loc['cutout_id']) for k in rows.keys()}
    _fc_params = make_false_color_params(cutouts)
    fc = lambda x: false_color(x, _fc_params)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3 * 2, 3 * 2))
    for row_i, CNN_class in enumerate(['TP', 'FN']):
        for col_i, MF_class in enumerate(['TP', 'FN']):
            key = 'MF %s & CNN %s' % (MF_class, CNN_class)
            cplt, r = axes[row_i, col_i], rows[key]
            cplt.imshow(fc(cutouts[key]),
                        extent=[cutout_size / 2, -cutout_size / 2, cutout_size / 2, -cutout_size / 2])
            cplt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            if not pd.isnull(r['CL_ra']):
                cplt.scatter([r['CL_ra'] - r['cutout_ra']], [r['CL_dec'] - r['cutout_dec']], s=200, c='red', marker='x', linewidth=3)
            if col_i == 0: cplt.set_ylabel('CNN %s'%CNN_class)
            if row_i == 0: cplt.set_title('MF %s' % MF_class)
            cplt.set_xlabel("S/N=%.2f   CNN Prob=%.2f\n Mvir=%.2e $M_\\odot$"%(r['pred_MF'], r['pred'], r['Mvir']), fontsize=font-4)
    if save_path is not None: plt.savefig(save_path, dpi=500, bbox_inches="tight")
    #plt.show()

def show_examples_breakchannels(df, title, get_x_func, n=5, feature='redshift', use_log=False,
                              font=DEFAULT_FONT, save_path=None, additional_df=None):
    #df should be filterd already
    print("%d cutouts are '%s'"%(len(df), _translate_one(title)))
    print("Below are examples whose %s are at different percentiles (shown on the left side of the left-most column) within these %d cutouts."%(feature, len(df)))
    print("%s increases from top to bottom"%feature)
    #print("Also showing all cutouts information sequntially below:\n")
    percs = np.linspace(0., 1., n + 1)[:n] + 0.5 / n
    cols = ['90kHz', '148kHz', '219kHz', 'false color combined']
    ncols = 4
    rows = ["%.2f%%-tile" % (p * 100) for p in percs]
    fig, axes = plt.subplots(nrows=len(rows), ncols=ncols, figsize=(3 * ncols, 3 * len(rows)))
    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, rotation=90, size='large')
    cutout_size = 8. / 60
    cutouts, rs = {}, {}
    for row_i, row in enumerate(rows):
        curr_percentile_val = df[feature].quantile(percs[row_i])
        r = df.loc[df[feature].map(lambda x: (x - curr_percentile_val) ** 2).idxmin()]
        print("Cutout {} has features: Mvir={:.1e} redshift={} rvir={}".format(r['cutout_id'], r['Mvir'], r['redshift'], r['rvir']))
        cutouts[row_i], rs[row_i] = get_x_func(r['cutout_id']), r
    if use_log:
        fc = lambda x: false_color_log(x)
        sc = lambda x: single_channel_log(x)
    else:
        print("When not using log scale (now), all false coloring in the same matrix is on the same scale")
        _fc_params = make_false_color_params(cutouts)
        fc = lambda x: false_color(x, _fc_params)
        sc = lambda x: x
    for row_i, row in enumerate(rows):
        x, r = cutouts[row_i], rs[row_i]
        for col_i, col in enumerate(cols):
            cplt = axes[row_i, col_i]
            if col_i == 3:
                cplt.imshow(fc(x), extent=[cutout_size/2, -cutout_size/2, cutout_size/2, -cutout_size/2])
            else:
                _min = x[:,:,col_i].min()
                im = cplt.imshow(sc(x[:,:,col_i]), extent=[cutout_size/2, -cutout_size/2, cutout_size/2, -cutout_size/2], cmap='gray')
                cbar = plt.colorbar(im, ax=cplt)
                if _min -1 > 0:
                    cbar.ax.set_ylabel("ln(x+%.2f)"%(_min - 1))
                else:
                    cbar.ax.set_ylabel("ln(x%.2f)" % (_min - 1))
            cplt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False,
                             left=False, labelleft=False)
            if not pd.isnull(r['CL_ra']):
                cplt.scatter([r['CL_ra'] - r['cutout_ra']], [r['CL_dec'] - r['cutout_dec']], s=400,
                             edgecolors='red' if col_i != 3 else 'black',
                             marker='o', facecolors='none')
                if additional_df is not None:
                    tdf = additional_df[additional_df['cutout_id'] == r['cutout_id']].dropna(subset=['CL_ra'])
                    if len(tdf) > 1:
                        for _idx in tdf.index:
                            _r = tdf.loc[_idx]
                            if _r['halo_id'] == r['halo_id']: continue
                            cplt.scatter([_r['CL_ra'] - _r['cutout_ra']], [_r['CL_dec'] - _r['cutout_dec']], s=200,
                                         edgecolors='red' if col_i != 3 else 'black',
                                         marker='o', facecolors='none')

            if row_i == 0: cplt.set_title(col, fontsize=font)
            if feature in ['rvir' ,'redshift', 'Mvir', 'tSZ']:
                cplt.set_xlabel("%s=%.2E" % (feature, r[feature]))
            else:
                cplt.set_xlabel("%s=%f" % (feature, r[feature]) + ("" if  pd.isnull(r["Mvir"]) else ", Mass=%.2E" % (r["Mvir"]) ))

    fig.tight_layout()
    if save_path is not None: plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.show()

def show_range_by_feature_sep(df, pred_cols, get_x_func, n=5, feature='redshift',
                           ycol='y', save_path=None):
    import matplotlib.pyplot as plt
    print("There are %d such cutouts" % (len(df)))
    percs = np.linspace(0., 1., n + 1)[:n] + 0.5 / n
    cols, col2idxs = [], {}
    for k in pred_cols.keys():
        if isinstance(pred_cols[k], str): pred_cols[k] = df[pred_cols[k]]
        cols.extend(["%s_TP" % k, "%s_FN" % k])
        col2idxs['%s_TP' % k] = df[df[ycol] & (pred_cols[k])].index
        col2idxs['%s_FN' % k] = df[df[ycol] & (~pred_cols[k])].index
        if feature not in {"redshift", "tSZ", "Mvir", "rvir"}:
            cols.extend(["%s_FP" % k, "%s_TN" % k])
            col2idxs['%s_FP' % k] = df[(~df[ycol]) & pred_cols[k]].index
            col2idxs['%s_TN' % k] = df[(~df[ycol]) & ~pred_cols[k]].index
        # else:
        # cols.extend(["%s_FP" % k, "%s_TN" % k])
    old_cols = cols.copy()
    cols = []
    for k in old_cols:
        if len(col2idxs[k]) > 0:
            cols.append(k)
        else:
            col2idxs.pop(k)
    rows = ["%.2f%%-tile" % (p * 100) for p in percs]
    import ipdb
    # ipdb.set_trace()
    fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), figsize=(3 * len(cols), 3 * len(rows)))

    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, rotation=90, size='large')
    cutout_size = 8. / 60
    cutouts, rs = {}, {}
    for row_i, row in enumerate(rows):
        for col_i, col in enumerate(cols):
            # print(row_i, col_i)
            tdf = df.reindex(col2idxs[col])
            curr_percentile_val = tdf[feature].quantile(percs[row_i])
            r = tdf.loc[tdf[feature].map(lambda x: (x - curr_percentile_val) ** 2).idxmin()]
            print("Cutout {} has features: Mvir={:.1e} redshift={} rvir={}".format(r['cutout_id'], r['Mvir'],
                                                                                   r['redshift'],
                                                                                   r['rvir']))
            cutouts[(row_i, col_i)], rs[(row_i, col_i)] = get_x_func(r['cutout_id']), r
    _fc_params = make_false_color_params(cutouts)
    fc = lambda x: false_color(x, _fc_params)
    for row_i, row in enumerate(rows):
        for col_i, col in enumerate(cols):
            # print(row_i, col_i)
            r = rs[(row_i, col_i)]
            axes[row_i, col_i].imshow(fc(cutouts[(row_i, col_i)]),
                                      extent=[cutout_size / 2, -cutout_size / 2, cutout_size / 2, -cutout_size / 2])
            axes[row_i, col_i].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False,
                                           right=False, left=False, labelleft=False)
            if not pd.isnull(r['CL_ra']):
                axes[row_i, col_i].scatter([r['CL_ra'] - r['cutout_ra']], [r['CL_dec'] - r['cutout_dec']], s=200,
                                           edgecolors='red', marker='o', facecolors='none')
            if feature in ['rvir', 'redshift', 'Mvir', 'tSZ']:
                axes[row_i, col_i].set_xlabel("%s=%.2E" % (feature, r[feature]))
            else:
                axes[row_i, col_i].set_xlabel("%s=%f" % (feature, r[feature]) + ", Mass=%.2E" % ("Mvir", r["Mvir"]))

    fig.tight_layout()
    if save_path is not None: plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.show()

MARKER_STYLE = dict(color='tab:blue', linestyle=':', marker='o',
                    markersize=15, markerfacecoloralt='tab:red')


def show_range_by_pred_value(df, get_x_func, nrow=4, ncol=4, feature='redshift', save_path=None, extra_info=[]):
        import matplotlib.pyplot as plt
        print("There are %d such cutouts, ordered by %s"%(len(df), feature))
        n = nrow * ncol
        percs = np.linspace(0., 1., n + 1)[:n] + 0.5 / n
        import ipdb
        # ipdb.set_trace()
        fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(3 * ncol, 3 * nrow))

        def _format_row(r):
            ss = "%s=%.2f"%(feature, r[feature])
            for _f in extra_info:
                ss += "\n%s=%.2f"%(_f, r[_f])
            return ss
        def _format_row_halo(r):
            #ss = "log(M)=%.2f, log(tSZ)=%.3f, z=%.2f"%(np.log(r['Mvir'])/np.log(10), np.log(r['tSZ'])/np.log(10), r['redshift'])
            ss = "Mass=%.2E, tSZ=%.2E, z=%.2f" % (r['Mvir'], r['tSZ'], r['redshift'])
            return ss


        cutout_size = 8. / 60
        cutouts = {}
        rs = {}
        for row_i in range(nrow):
            for col_i in range(ncol):
                tdf = df.reindex()
                curr_percentile_val = tdf[feature].quantile(percs[ncol * row_i + col_i])
                r = tdf.loc[tdf[feature].map(lambda x: (x - curr_percentile_val) ** 2).idxmin()]
                cutouts[(row_i, col_i)] = get_x_func(r['cutout_id'])
                rs[(row_i, col_i)] = r
        _fc_params = make_false_color_params(cutouts)
        fc = lambda x: false_color(x, _fc_params)
        for row_i in range(nrow):
            for col_i in range(ncol):
                r = rs[(row_i, col_i)]
                axes[row_i, col_i].imshow(fc(cutouts[(row_i, col_i)]), extent=[cutout_size/2, -cutout_size/2, cutout_size/2, -cutout_size/2])
                axes[row_i, col_i].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
                if not pd.isnull(r['CL_ra']):
                    axes[row_i, col_i].scatter([r['CL_ra'] - r['cutout_ra']], [r['CL_dec'] - r['cutout_dec']], s=200, edgecolors='red', marker='o', facecolors='none')#, linewidth='3')
                axes[row_i, col_i].set_xlabel(_format_row(r))
                axes[row_i, col_i].set_title(_format_row_halo(r))

        fig.tight_layout()
        if save_path is not None: plt.savefig(save_path, dpi=500, bbox_inches="tight")
        plt.show()

def plot_features(df, x='redshift', y='rvir', c='tSZ', font=DEFAULT_FONT, ranges={}):
    plt.rcParams.update({'font.size': font})
    col_to_name = {"rvir": "Virial Radius (Mpc)", "Mvir": "Virial Mass ($M_\\odot$)",
                   "tSZ":"tSZ (arcmin^2)", "redshift":"Redshift"}
    cm = plt.cm.get_cmap('RdYlBu')
    if c in ranges:
        sc = plt.scatter(df[x], df[y], c=df[c], cmap=cm, vmax=ranges[c][1], vmin=ranges[c][0])
    else:
        sc = plt.scatter(df[x], df[y], c=df[c], cmap=cm)
    ylab = col_to_name.get(y,y)
    plt.ylabel(ylab)
    if y in ranges: plt.ylim(*ranges[y])

    xlab = col_to_name.get(x,x)
    plt.xlabel(xlab)
    if x in ranges: plt.xlim(*ranges[x])
    # legend
    cbar = plt.colorbar(sc)
    #cbar.set_label(c, rotation=270, horizontalalignment='right')
    cbar.ax.set_title(col_to_name.get(c,c), fontsize=font - 4)
    #cbar.set_title(col_to_name.get(c,c), rotation=270, labelpad=14)
    # plt.show()
    pass


def plot_relative_loc(df, cutout_size=8./60, font=DEFAULT_FONT):
    df['rel_ra'] = df['CL_ra'] - df['cutout_ra']
    df['rel_dec'] = df['CL_dec'] - df['cutout_dec']
    plt.scatter(df['rel_ra'], df['rel_dec'])
    plt.xlabel("Relative RA", fontsize=font)
    plt.xlim((-cutout_size / 2, cutout_size/2))
    plt.ylabel("Relative Dec", fontsize=font, labelpad=0)
    plt.ylim((-cutout_size / 2, cutout_size / 2))
    #plt.title("Halo dist. within cutout")


def plot_all(df, cutout_size=8./60,save_path=None, font=DEFAULT_FONT, ranges = {}):
    # features, relative locations,
    f = plt.figure(figsize=(16, 5))
    plt.rcParams.update({'font.size': font})

    """
    fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), figsize=(3 * len(cols), 3 * len(rows)))
    fig.tight_layout()
    """
    f.add_subplot(1, 2, 1)
    plot_features(df, font=font, ranges=ranges)
    #plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    f.add_subplot(1, 2, 2)

    plot_relative_loc(df, cutout_size=cutout_size)
    f.tight_layout()
    if save_path is not None: plt.savefig(save_path, dpi=500, bbox_inches="tight")
    pass


CB_color_cycle = [
                #'377eb8',
                #'ff7f00',
                #'4daf4a',
                #'f781bf',
                #'a65628',

                #'984ea3',
                #'999999',
                #'e41a1c',
                'crimson',
                'cyan',
                'silver',
                    'peru',
    'dodgerblue',
    'purple',
    'black'
                    ]
def plot_training_process(result, ratio_map, key='acc', save_path=None, font=DEFAULT_FONT):
    plt.rcParams.update({'font.size': font})
    colors = CB_color_cycle.copy()
    full_spelling = {"acc":"Accuracy", "acc_adj":"Accuracy Adjusted", "loss": "Loss"}

    ratio_map_raw = ratio_map.copy()
    ratio_map = pd.Series(index=range(max(ratio_map_raw.values()) + 1))
    for k in sorted(ratio_map_raw.keys()):
        start_idx = 0 if k == 0 else ratio_map_raw[k - 1]
        ratio_map[(ratio_map.index < ratio_map_raw[k]) & (ratio_map.index >= start_idx)] = k + 1
    if key.startswith("acc"):
        acc_adj = key.endswith("adj")
        key = 'acc'
    val_ser = pd.DataFrame(result['val'])[key].sort_index()
    if key == 'F1':
        df = pd.DataFrame({"train": np.NaN, 'val': val_ser})
    else:

        train_ser = pd.Series(result['train'][key]).sort_index()
        window = (val_ser.index[1] - val_ser.index[0]) // (train_ser.index[1] - train_ser.index[0])
        print("Rolling_window = {}".format(window))
        df = pd.DataFrame({"train": train_ser.rolling(window).mean().dropna(), 'val': val_ser})
        if key == 'acc':
            adj_ser = df.index.map(lambda x: 1 - 1. / (1 + ratio_map[x]))
            if acc_adj:
                for c in df.columns: df[c] = df[c] - adj_ser

    f = plt.figure(figsize=(14, 5))
    #fig, ax1 = plt.subplots()
    #f.add_subplot(2, 1, 1)
    lns = []
    for label in ['train', 'val']:
        lns.extend(plt.plot(df[label].dropna().index, df[label].dropna().values,
                            label="%s %s"%({"train":"Train", "val":"Valid"}[label],full_spelling[key]), color=colors.pop(),
                            linewidth=3))
    if key == 'acc':
        if acc_adj:
            colors.pop()
        else:
            lns.extend(plt.plot(df.index, adj_ser, color=colors.pop(), label='Blind Guess',
                                linewidth=3))
    else:
        colors.pop()
    #plt.xlabel("Total Batches", fontsize=font)
    #plt.ylabel(full_spelling[key], fontsize=font)
    plt.xlabel("Total Batches")
    plt.ylabel(full_spelling[key])
    ax2 = plt.twinx()
    ax2.set_ylabel("Negative-to-Positive Ratio",fontsize=font, rotation=270, labelpad=17)
    lns.extend(ax2.plot(ratio_map, label='Negative-to-Positive Ratio',color=colors.pop(), linewidth=3, linestyle='--'))

    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='lower right' if (key=='acc' and not acc_adj) else 'right')#, prop={"size":font})

    if save_path is not None: plt.savefig(save_path, dpi=500, bbox_inches="tight")

    return

def plot_rocs(y, preds, save_path=None, font=DEFAULT_FONT):
    plt.rcParams.update({'font.size': font})
    colors = CB_color_cycle.copy()
    fprs = {}
    tprs = {}
    aucs = {}
    cm_norm2 = {}
    plt.figure(figsize=[12, 8])
    for k in preds.keys():
        _ypred = preds[k]
        fprs[k], tprs[k], _ = metrics.roc_curve(y, _ypred, pos_label=1)
        aucs[k] = metrics.roc_auc_score(y, _ypred)
        plt.plot(fprs[k], tprs[k], color=colors.pop(), lw=3, label="%s AUC=%.2f"%(k, aucs[k]))
    plt.plot([0, 1], [0, 1], color='k', lw=1, linestyle='-')
    plt.xlim([0.0, 0.3])
    plt.ylim([0.7, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC', fontsize=20)
    plt.legend(loc="lower right")
    plt.tick_params(axis='both', which='major')
    plt.grid()
    if save_path is not None: plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.show()

def plot_F1s(y, preds, xlim=(0.2, 0.8), save_path=None, font=DEFAULT_FONT,
             y_test=None, pred_test=None):
    plt.rcParams.update({'font.size': font})
    colors = CB_color_cycle.copy()
    plt.figure(figsize=[12, 8])
    ts = {}
    for k in preds.keys():
        _ypred = preds[k]
        _x, _y = get_F1(_ypred, y, plot=False, xlim=xlim, get_raw=True)
        _t, _ = get_F1(_ypred, y, plot=False, xlim=xlim, get_raw=False)
        ts[k] = _t
        plt.plot(_x, _y, color=colors.pop(), lw=3, label="%s peak=%.2f" % (k, np.max(_y)))
        if y_test is not None: print(f"{k} - F1:{_get_Fbeta(y_test, pred_test[k] > _t, debug=True)}, thres={_t}")
    plt.xlim(xlim)
    plt.xlabel('CNN Prob')
    plt.ylabel('F1')
    plt.legend(loc="lower center")
    if save_path is not None: plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.show()
    return ts


def plot_edge_effects_1(df, dist_thress = np.linspace(0., 0.02, 21), save_path=None):
    colors = CB_color_cycle.copy()
    plt.figure(figsize=[12, 8])
    neg2pos_ratio = len(df[~df['y']]) / float(len(df[df['y']]))
    result_df = pd.DataFrame(index=dist_thress, columns=['precision', 'recall', 'F1', 'Npos'])
    for dist_thres in dist_thress:
        tdf = df[(~(df['pos_dist2edge'] < dist_thres)) & (~(df['neg_dist2edge'] < dist_thres))]

        n_drop_negs = len(tdf[~tdf['y']]) - int(tdf['y'].sum() * neg2pos_ratio)
        np.random.seed(0)
        tnandf = tdf[(~tdf['y']) & tdf['neg_dist2edge'].isnull()]
        tdropdf = tnandf.reindex(tnandf.index[np.random.permutation(len(tnandf))][:n_drop_negs])
        # print(tdf.shape, tnandf.shape, tdropdf.shape)
        tdf = tdf.reindex(tdf.index.difference(tdropdf.index))
        # print(len(tdf[~tdf['y']]) / float(len(tdf[tdf['y']])))
        # tdf = tdf[tdf['']]
        new_val_thres, _ = get_F1(tdf[tdf['which'] == 'valid']['pred_%s' % "skymap"],
                                  tdf[tdf['which'] == 'valid']['y'], plot=False)

        result_df.loc[dist_thres, :3] = _get_Fbeta(tdf[tdf['which'] == 'test']['y'],
                                                  tdf[tdf['which'] == 'test']['pred_skymap'] > new_val_thres,
                                                  get_raw=True)
        result_df.loc[dist_thres, 'Npos'] = tdf['y'].sum()
    lns = []
    for c in result_df.columns[:3]:
        lns.extend(plt.plot(result_df.index, result_df[c].values, color=colors.pop(), lw=3, label=c))
        plt.xlabel('threshold for distance from center to be kept', fontsize=16)
        #plt.ylabel('F1', fontsize=16)
        # plt.title('ROC', fontsize=20)


    ax2 = plt.twinx()
    ax2.set_ylabel("Num of positives", fontsize=14)
    lns.extend(ax2.plot(result_df.index, result_df['Npos'], label='# of positives', color=colors.pop()))
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='upper center', prop={"size": 14})
    # plt.tick_params(axis='both', which='major', labe
    if save_path is not None: plt.savefig(save_path, dpi=500)
    plt.show()
    return result_df


def plot_pred_vs_dist_with_errorbar(df, save_path=None, ngroups=11, quantiles=(0.1, 0.9), font=DEFAULT_FONT):
    plt.rcParams.update({'font.size': font, 'legend.fontsize': font-5})
    df['dist'] = df['ratio'] * 3. #to arcmin
    ids = df['halo_id'].unique()
    groups = np.array_split(np.asarray(ids), ngroups)
    res = {}
    for i, curr_group in enumerate(groups):
        tser = df[df['halo_id'].isin(curr_group)].groupby('dist')['pred'].mean()
        res[i] = tser
    res = pd.DataFrame(res)
    df_to_plot = pd.DataFrame({"Mean":res.mean(axis=1),
                               "Quantile %.2f"%quantiles[0]: res.quantile(quantiles[0], axis=1),
                               "Quantile %.2f"%quantiles[1]: res.quantile(quantiles[1], axis=1)})
    ax = df_to_plot.plot()
    ax.set_ylim((0, 0.9))
    ax.set_xlabel('Distance from Center (arcmin)')
    ax.set_ylabel('Average CNN Score')
    if save_path is not None: plt.savefig(save_path, dpi=500, bbox_inches="tight")

def plot_all_features_pairs(df, save_path=None, font=DEFAULT_FONT, ylabel=None):
    df = df.reindex().set_index('cutout_id')
    df['Mvir'] = df['Mvir'].map(np.log) / np.log(10)
    df['tSZ'] = df['tSZ'].map(np.log) / np.log(10)
    truth = None if ylabel is None else df[ylabel]
    if truth is not None:
        TP_df, FP_df = df[truth], df[~truth]
    col_to_name = {"rvir": "virial radius", "Mvir": "log_10(virial mass)", "redshift":"redshift", "tSZ":"log_10(tSZ)"}
    assert all([df[list(col_to_name.keys())[i]].count() == df[list(col_to_name.keys())[i-1]].count() for i in range(1, len(col_to_name))])
    print("Size is %d"%df[list(col_to_name.keys())[0]].count())
    #outliers_idx = df['tSZ'].sort_values(inplace=False).dropna().tail(2).index
    #print("There are two outliers for tSZ: {}. For readability, removing these two clusters.".format(df['tSZ'].reindex(outliers_idx)))
    #df = df.reindex(df.index.difference(outliers_idx))

    assert all([c in df.columns for c in col_to_name.keys()])
    nc = len(col_to_name)
    #f = plt.figure(figsize=(4 * nc, 4 * nc))
    fig, axes = plt.subplots(nrows=nc, ncols=nc, figsize=(5 * nc, 5 * nc))
    for xi, xc in enumerate(col_to_name.keys()):
        #each column share the same x
        for yi, yc in enumerate(col_to_name.keys()):
            colors = CB_color_cycle.copy()
            cplt = axes[yi, xi]
            if yc == xc:
                _, bins, _ = cplt.hist(df[yc], color=colors.pop(), label='FP')
                if truth is not None:
                    cplt.hist(TP_df[yc], bins=bins, color=colors.pop(), label='TP')
                cplt.set_xlabel("hist of %s" % (col_to_name[xc]), fontsize=int(font /1))
            else:
                if truth is not None:
                    cplt.scatter(FP_df[xc], FP_df[yc], label='FP', color=colors.pop(), alpha=0.5)
                    cplt.scatter(TP_df[xc], TP_df[yc], label='TP', color=colors.pop(), alpha=0.5)
                else:
                    cplt.scatter(df[xc], df[yc], color=colors.pop(), alpha=0.5)
            if truth is not None and xi == 0: cplt.legend()
            if xi == 0: cplt.set_ylabel(col_to_name[yc], fontsize=font)

            if yi == 0: cplt.set_title(col_to_name[xc], fontsize=font)
    if save_path is not None: plt.savefig(save_path, dpi=500, bbox_inches="tight")


def plot_all_features_hist_by_component(dfs, save_path=None, font=DEFAULT_FONT):
    nrows = len(dfs)
    col_to_name = {"rvir": "virial radius", "Mvir": "log_10(virial mass)", "redshift": "redshift", "tSZ": "log_10(tSZ)"}
    nc = len(col_to_name)
    #f = plt.figure(figsize=(4 * ro, 4 * nc))
    fig, axes = plt.subplots(nrows=nrows, ncols=nc, figsize=(3 * nc, 3 * nrows))
    for row_i, k in enumerate(dfs.keys()):
        df = dfs[k]
        df = df.reindex().set_index('cutout_id')
        df['Mvir'] = df['Mvir'].map(np.log) / np.log(10)
        df['tSZ'] = df['tSZ'].map(np.log) / np.log(10)

        assert all([df[list(col_to_name.keys())[i]].count() == df[list(col_to_name.keys())[i-1]].count() for i in range(1, len(col_to_name))])
        print("Size is %d"%df[list(col_to_name.keys())[0]].count())
        #outliers_idx = df['tSZ'].sort_values(inplace=False).dropna().tail(2).index
        #print("There are two outliers for tSZ: {}. For readability, removing these two clusters.".format(df['tSZ'].reindex(outliers_idx)))
        #df = df.reindex(df.index.difference(outliers_idx))

        assert all([c in df.columns for c in col_to_name.keys()])

        for xi, xc in enumerate(col_to_name.keys()):
            #each column share the same x
            axes[row_i, xi].hist(df[xc])
            #axes[row_i, xi].xlabel("hist of %s" % (col_to_name[xc]), fontsize=int(font / 1))
            if xi == 0: axes[row_i, xi].set_ylabel(k, fontsize=font)

            if row_i == 0: axes[row_i, xi].set_title(col_to_name[xc], fontsize=font)
        if save_path is not None: plt.savefig(save_path, dpi=500, bbox_inches="tight")

def _make_bins(x, log=False, nbins=50):
    if isinstance(x, list):
        return _make_bins(np.concatenate(x), log=log, nbins=nbins)
    #ipdb.set_trace()
    x = x[~np.isnan(x)]
    v = np.log10(x) if log else x
    bins = np.linspace(np.min(v), np.max(v), nbins)
    bins = np.power(10, bins) if log else bins
    return bins



def _hist2d(ax, x, y, nbins=50, logx=False,logy=False):
    #nan_idx = np.isnan(x)
    #assert np.equal(nan_idx, np.isnan(y))
    #x = x[~nan_idx]
    #y = y[~nan_idx]
    xbins = _make_bins(x, logx, nbins)
    ybins = _make_bins(y, logy, nbins)
    counts, _, _ = np.histogram2d(x, y, bins=(xbins, ybins))
    counts = np.log(counts)
    ax.pcolormesh(xbins, ybins, counts.T, cmap='Greys')
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    return xbins, ybins

def _need_log(f):
    return f in {"Mvir", "tSZ"}

def _mid(bins):
    if bins[1] - bins[0] == bins[2] - bins[1]:
        return 0.5 * bins[1:] + 0.5 * bins[:-1]
    else:
        bins = np.log10(bins)
        return np.power(10, 0.5 * bins[1:] + 0.5 * bins[:-1])

def plot_all_features_corr_by_component_fullset_hist2d_proportion(df, idxs, save_path=None, font=DEFAULT_FONT, min_cutouts_per_pixel=1, nbins=50, pairs=None):
    assert isinstance(idxs, dict)
    plt.rcParams.update({'font.size': font})
    df = df.reindex().dropna(subset=['redshift'])
    ncol = len(idxs)
    col_to_name = {"rvir": "Virial Radius (Mpc)",
                   "angle":"Angular Size (arcmin)",
                   "Mvir": "Virial Mass ($M_\\odot$)",
                   "redshift": "Redshift",
                   "tSZ": "tSZ (arcmin^2)"}
    assert all([df[list(col_to_name.keys())[i]].count() == df[list(col_to_name.keys())[i - 1]].count() for i in
                range(1, len(col_to_name))])
    assert all([c in df.columns for c in col_to_name.keys()])
    #assert all([c in df.columns for c in show])
    if pairs is None:
        pairs = [(list(col_to_name.keys())[i], list(col_to_name.keys())[j]) for i in range(len(col_to_name)) for j in range(i+1, len(col_to_name))]
    nrow = len(pairs)
    #f = plt.figure(figsize=(4 * ro, 4 * nc))
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(6 * ncol, 5 * nrow))
    """
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(3, nc),
                     axes_pad=0.1,
                     share_all=True,
                     cbar_location="right",
                     cbar_size="4%",
                     cbar_pad=0.1,
                     cbar_mode='edge')
    """
    #for col_i, k in enumerate(thresholds.keys()):
    for col_i, subset in enumerate(sorted(idxs)):
        tdf = df[idxs[subset]]

        for xi, pair in enumerate(pairs):
            cplt = axes[xi, col_i] if ncol > 1 else axes[xi]
            xf, yf = pair
            logx = xf in {"Mvir", "tSZ"}
            logy = yf in {"Mvir", "tSZ"}
            xbins = _make_bins(df[xf], logx, nbins=nbins)
            ybins = _make_bins(df[yf], logy, nbins=nbins)
            ccounts, _, _ = np.histogram2d(tdf[xf], tdf[yf], bins=(xbins, ybins))
            totalcounts, _, _ = np.histogram2d(df[xf], df[yf], bins=(xbins, ybins))
            totalcounts[totalcounts < min_cutouts_per_pixel] = 0.

            sc = cplt.pcolormesh(_mid(xbins), _mid(ybins), ((ccounts * 100.0) / totalcounts).T, cmap='Greens', vmin=0., vmax=100.)

            cntr = cplt.contour(_mid(xbins), _mid(ybins), totalcounts.T, extend=[xbins.min(), xbins.max(), ybins.min(), ybins.max()], levels=[min_cutouts_per_pixel-1], colors='black')
            if logx: cplt.set_xscale("log")
            if logy: cplt.set_yscale("log")

            if xi == 0: cplt.set_title(subset)#, fontsize=font)
            if _need_log(xf): cplt.set_xscale("log")
            if _need_log(yf): cplt.set_yscale("log")
            if yf == 'tSZ': cplt.set_ylim((1e-10, 1e-1))

            if yf == 'Mvir' or xf == 'Mvir':
                getattr(cplt, 'axhline' if yf == 'Mvir' else 'axvline')(2e14, color='black', linestyle='--')
            cplt.set_xlabel(col_to_name[xf])#, fontsize=font)
            cplt.set_ylabel(col_to_name[yf])#, fontsize=font)

    fig.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.ax.set_title('% Positive', fontsize=font - 4)
    if save_path is not None: plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.show(block=True)

def plot_completeness(df, methods=['CNN', 'MF', 'EnsemblePROD'],
                      save_path=None, font=DEFAULT_FONT,
                      col='Mvir', q0=0.2, q1=1.):
    import matplotlib.ticker as ticker
    colors = CB_color_cycle.copy()
    col_name = {"Mvir": "Mass", "redshift": "Redshift"}[col]
    plt.rcParams.update({'font.size': font})
    all_vals = df[col].dropna()
    mmin, mmax = all_vals.quantile(q0), all_vals.quantile(q1)
    df = df[(df[col] > mmin) & (df[col] < mmax)]


    if col == 'Mvir':
        all_thres = np.logspace(np.log10(mmin), np.log10(mmax), 20)
    else:
        all_thres = np.linspace(mmin, mmax, 20)
    print(all_thres)
    get_cnt = lambda vals, thres=all_thres: pd.Series({m: vals[(vals >= m)&(vals <thres[i+1])].count() for i,m in enumerate(thres[:-1])})
    all_vals = df[col].dropna()
    df2 = df[(df['redshift'] > 0.25) if col == 'Mvir' else (df['Mvir'] > 2e14)]

    from matplotlib import gridspec
    fig = plt.figure(figsize=(20,10 * (4./3 if col == 'Mvir' else 1.)))
    #fig, ax = plt.subplots(figsize=(20,10))
    if col == 'Mvir':
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax0 = plt.subplot(gs[0])
    #ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=5))
    suffix = " (%s)"%("redshift>0.25" if col == 'Mvir' else "mass>2e14")
    lns, legends = [], []
    curves = {method: get_cnt(df[df['%s_pred'%method]][col].dropna()) / get_cnt(df[col].dropna()) for method in ['CNN', 'MF', 'EnsemblePROD']}
    curves.update({'%s%s'%(method, suffix): get_cnt(df2[df2['%s_pred'%method]][col].dropna()) / get_cnt(df2[col].dropna()) for method in ['CNN', 'MF', 'EnsemblePROD']})
    for method in methods:
        color = colors.pop()
        if col == 'Mvir':
            lns.extend(plt.plot(_mid(all_thres), curves[method], label=method, color=color))
            legends.append(method)
        lns.extend(plt.plot(_mid(all_thres), curves['%s%s'%(method, suffix)], label='%s%s'%(method, suffix), linestyle='dashed', color=color))
        legends.append('%s%s'%(method, suffix))
    plt.vlines(2e14 if col == 'Mvir' else 0.25, 0, 1, label='Threshold', color='red')
    plt.ylabel("Completeness (Recall)")
    plt.hlines(1,0,2e15 if col == 'Mvir' else 2,alpha=0.2)

    #lns.extend(plt.vlines(2e14, 0, 1, label='Threshold'))
    plt.xlabel(col_name)
    ax2 = plt.twinx()
    plt.hist(all_vals, bins=all_thres, log=True, alpha=0.4)
    plt.ylabel("Counts of Objects")
    if col == 'Mvir':
        y_labels = []
    else:
        y_labels = []
    #ax2.set_yticklabels(y_labels)
    plt.legend(lns, legends, loc='right')
    #plt.title('Completeness (Recall) vs %s' % col_name)
    print('Completeness (Re call) vs %s' % col_name)

    if col == 'Mvir':
        plt.xscale('log')
        #cplt = fig.add_axes([0.12, -0.2, 0.78, 0.2])
        cplt = plt.subplot(gs[1], sharex=ax0)
        colors = CB_color_cycle.copy()
        assert methods == ['CNN', 'MF', 'EnsemblePROD'], "This is only for the default methods"
        lns2, legends2 = [], []
        for method in methods[:-1]:
            color = colors.pop()
            lns2.extend(cplt.plot(_mid(all_thres)[-12:], (curves[method] / curves['EnsemblePROD']).iloc[-12:], label='%s/EnsemblePROD'%method, color=color))
            lns2.extend(cplt.plot(_mid(all_thres)[-12:], (curves['%s%s'%(method, suffix)] / curves['EnsemblePROD']).iloc[-12:], label='%s/EnsemblePROD%s'%(method, suffix), color=color, linestyle='dashed'))
            legends2.extend(["%s/EnsemblePROD"%method, '%s/EnsemblePROD%s'%(method, suffix)])

        plt.setp(ax0.get_xticklabels(), visible=True)
        plt.legend(lns2, legends2, loc='upper left', fontsize=10)
        #plt.setp(cplt.get_xticklabels(), visible=False)
        plt.subplots_adjust(hspace=.0)
        cplt.set_xscale('log')
        #cplt.xaxis.tick_top()
        cplt.xaxis.set_label_position('bottom')
        cplt.set_xlabel(col_name + ' ($M_\\odot$)')
        #cplt.set_xlim(ax0.get_xlim())
        #plt.ylabel("As a % of EnsemblePROD")


    if save_path is not None: plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.show(block=True)
    return curves




def plot_all_hists_by_features(df, thresholds, save_path=None, font=DEFAULT_FONT, fullset='truth', nbins=10):
    print("the height of the black bar is the all positives (TP+FN), so the unmasked portion is TP")
    df = df.reindex()
    assert fullset in{"pred", "truth"}

    nrows = len(thresholds)
    col_to_name = {"rvir": "virial radius", "Mvir": "log_10(virial mass)", "redshift": "redshift", "tSZ": "log_10(tSZ)"}
    assert all([df[list(col_to_name.keys())[i]].count() == df[list(col_to_name.keys())[i - 1]].count() for i in
                range(1, len(col_to_name))])
    assert all([c in df.columns for c in col_to_name.keys()])
    ncols = len(col_to_name)
    #f = plt.figure(figsize=(4 * ro, 4 * nc))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    for row_i, k in enumerate(thresholds.keys()):
        #print("Size of dataframe is %d"%df[list(col_to_name.keys())[0]].count())

        pred_y = df['pred_%s'%k] > thresholds[k]
        TP_df = df[(pred_y) & df['y']]

        for xi, feature in enumerate(col_to_name.keys()):
            cplt = axes[row_i, xi]
            colors = CB_color_cycle.copy()
            #each column share the same x
            bins = _make_bins(df[feature], _need_log(feature), nbins=nbins)
            if fullset == 'pred':
                cplt.hist(df[pred_y][feature], bins=bins, color=colors.pop(), label='FP')
            else:
                cplt.hist(df[df['y']][feature], bins=bins, color=colors.pop(), label='FN')
            cplt.hist(TP_df[feature], bins=bins, color=colors.pop(), label='TP')
            #_, bins, _ = cplt.hist(df[feature], color= colors.pop(), label='TP')
            #cplt.hist(FN_df[feature], bins=bins, color=colors.pop(), label='FN')
            if xi == 0 and row_i == 0: cplt.legend()
            #axes[ xi, col_i].scatter(TP_df[xf], TP_df[yf], color=colors.pop(), label='TP', alpha=0.5)
            #axes[xi, col_i].scatter(FN_df[xf], FN_df[yf], color=colors.pop(), label='FN', alpha=0.5)
            #if xi == 0 and col_i == 0: axes[xi, col_i].legend()

            if row_i == 0: cplt.set_title(col_to_name[feature], fontsize=font)
            if xi == 0: cplt.set_ylabel(k, fontsize=font)
            if _need_log(feature): cplt.set_xscale("log")
        if save_path is not None: plt.savefig(save_path, dpi=500, bbox_inches="tight")


def plot_all_hists_by_features_stack_components(df, thresholds, save_path=None, font=DEFAULT_FONT, nbins=6):
    print("Where the bars sit is the middle point of the interval")
    assert df['y'].sum() == len(df)
    print("These are all positive samples")
    df = df.reindex()
    #df['Mvir'] = df['Mvir'].map(np.log) / np.log(10)
    #df['tSZ'] = df['tSZ'].map(np.log) / np.log(10)

    #bins = {"Mvir": np.linspace}
    #nrows = len(thresholds)
    nrows = 1
    col_to_name = {"rvir": "virial radius", "Mvir": "virial mass", "redshift": "redshift", "tSZ": "tSZ"}
    assert all([df[list(col_to_name.keys())[i]].count() == df[list(col_to_name.keys())[i - 1]].count() for i in
                range(1, len(col_to_name))])
    assert all([c in df.columns for c in col_to_name.keys()])
    ncols = len(col_to_name)
    #f = plt.figure(figsize=(4 * ro, 4 * nc))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    for xi, feature in enumerate(col_to_name.keys()):
        cplt = axes[xi]
        all_xs = [df[df['pred_%s'%k] > thresholds[k]][feature] for k in thresholds.keys()]
        bins = _make_bins(all_xs, _need_log(feature), nbins=nbins)
        cplt.hist(all_xs, bins=bins, label=list(thresholds.keys()), color=CB_color_cycle.copy())
        #print("Bins:{}".format(bins))

        cplt.legend()
        cplt.set_title(col_to_name[feature], fontsize=font)
        if _need_log(feature): cplt.set_xscale("log")
    return




def _translate_one(x):
    if x == 'true P': return "true positves"
    if x == 'All': return "all cutouts"
    method, _set = x.split(" ")
    if method not in {"CNN", "MF", "EnsembleAND", "EnsemblePROD", "MFInAND", "CNNInAND"}:
        method = "CNN on +%s component"%method
    if _set == "P": return "predicted-positives (TP+FP) by %s"%method
    return "%s by %s"%(_set, method)
def _translate_sets(l):
    return [_translate_one(x) for x in l]

def plot_all_hists_by_features_stack_general(df, save_path=None, font=DEFAULT_FONT, nbins=6, sets=[{}],
                                             handle_all='axis', rescale=True, percent=False):
    assert handle_all in {"axis"}, "{} is not implemented".format(handle_all)
    print("Where the bars sit is the middle point of the interval")
    print("\n\nFrom left to right, we have\n ({})".format("),\n(".join([" v.s. ".join(_translate_sets(x.keys())) for x in sets])))

    #assert df['y'].sum() == len(df)
    #print("These are all positive samples")
    df = df.reindex()
    ncols = len(sets)
    col_to_name = {"rvir": "virial radius", "Mvir": "virial mass", "redshift": "redshift", "tSZ": "tSZ"}
    print("\n\nFrom top to bottom, we have the histogram of {}".format(", ".join(col_to_name.values())))
    assert all([df[list(col_to_name.keys())[i]].count() == df[list(col_to_name.keys())[i - 1]].count() for i in
                range(1, len(col_to_name))])
    assert all([c in df.columns for c in col_to_name.keys()])
    nrows = len(col_to_name)
    #f = plt.figure(figsize=(4 * ro, 4 * nc))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    for col_i, csets in enumerate(sets):
        for xi, feature in enumerate(col_to_name.keys()):
            lns = []
            cplt = axes[xi, col_i] if ncols > 1 else axes[xi]
            good_ks = [_k for _k in csets.keys() if _k != 'All']
            labels = good_ks.copy()
            all_xs = [df[csets[_k]][feature] for _k in good_ks]
            if 'All' in csets.keys():
                bins = _make_bins(df[csets['All']][feature], _need_log(feature), nbins=nbins)
            else:
                bins = _make_bins(all_xs, _need_log(feature), nbins=nbins)
            if percent:
                hists = [np.histogram(_x, bins=bins)[0].astype(float) for _x in all_xs]
                total = sum(hists)
                hists = [_x/total for _x in hists]
                _mid_bins = _mid(bins)
                colors = CB_color_cycle.copy()
                for i in range(len(all_xs)):
                    lns.append(cplt.bar(_mid_bins, hists[i], color=colors.pop(), width = 0.3 * (bins[1:]-bins[:-1]), bottom=hists[i-1] if i > 0 else 0, label=labels))
                cplt.set_ylim(0, 1.4)
                cplt2 = cplt.twinx()
                lns.append(cplt2.bar(_mid_bins, total, label='Sum', alpha=0.3, color=colors.pop(), width = 0.6 * (bins[1:]-bins[:-1])))
                labels += ['Sum']
            else:
                lns.extend(cplt.hist(all_xs, bins=bins, label=labels, color=CB_color_cycle.copy()[:len(all_xs)])[2])
            if len(all_xs) == 1: lns = [lns[0]]
            #print("Bins:{}".format(bins))
            if 'All' in csets.keys():
                cplt2 = cplt.twinx() if rescale else cplt
                lns.extend(cplt2.hist(df[csets['All']][feature], bins=bins, label='All', alpha=0.3, color=CB_color_cycle[-1])[2][:1])
                labels.append('All' + (" (right axis)" if rescale else ""))

            #cplt.legend()
            cplt.legend(lns, labels)
            if col_i == 0:  cplt.set_ylabel(col_to_name[feature], fontsize=font)
            if _need_log(feature): cplt.set_xscale("log")
    return

def plot_perfmormance_by_feature_bins(df, FULL_idxmaps, save_path=None, font=DEFAULT_FONT, nbins=6, methods=['CNN', 'MF'], fp=False):
    #print("Where the bars sit is the middle point of the interval")
    print("Black vertical line is the mass threshold")
    #print("\n\nFrom left to right, we have\n ({})".format("),\n(".join([" v.s. ".join(_translate_sets(x.keys())) for x in sets])))

    df = df.reindex()
    col_to_name = {"rvir": "virial radius", "Mvir": "virial mass", "redshift": "redshift", "tSZ": "tSZ"}
    #print("\n\nFrom top to bottom, we have the histogram of {}".format(", ".join(col_to_name.values())))
    assert all([df[list(col_to_name.keys())[i]].count() == df[list(col_to_name.keys())[i - 1]].count() for i in
                range(1, len(col_to_name))])
    assert all([c in df.columns for c in col_to_name.keys()])
    nrows = len(col_to_name)
    #f = plt.figure(figsize=(4 * ro, 4 * nc))
    ncols = len(methods)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    for col_i, method in enumerate(methods):
        for xi, feature in enumerate(col_to_name.keys()):
            lns = []
            colors = CB_color_cycle.copy()
            cplt = axes[xi, col_i] if ncols > 1 else axes[xi]
            filter_idx = df.index if feature != 'Mvir' else ((df['Mvir'] <2e14) if fp else (df['Mvir'] > 2e14))
            bins = _make_bins(df.loc[filter_idx][feature], _need_log(feature), nbins=nbins)
            _width, _mid_bins = (bins[1:] - bins[:-1]), _mid(bins)

            if method == 'All':
                cplt.hist(df[FULL_idxmaps['All']][feature], bins=bins, label='All', alpha=0.4)
                #lns.append(cplt.bar(_mid_bins, bars['All'], label='All', alpha=0.2, color=colors[len(to_plot_bars)], width=0.6 * _width))
                cplt.set_yscale('log')
                cplt.legend()
            else:
                all_xs = {_k.split(" ")[1]: df[FULL_idxmaps[_k]].loc[filter_idx][feature] for _k in FULL_idxmaps.keys() if _k.startswith(method)}
                bars = {_k: np.histogram(all_xs[_k], bins=bins)[0].astype(float) for _k in all_xs.keys()}

                if fp:
                    to_plot_bars = {"FP_rate": bars['FP'] / (bars['FP'] + bars['TN'])}
                    cplt.set_yscale('log')
                else:
                    to_plot_bars = {"precision": bars['TP'] / bars['P'], "recall":bars['TP'] / (bars['FN'] + bars['TP'])}
                    to_plot_bars['F1'] = 2 * to_plot_bars['precision'] * to_plot_bars['recall'] / (to_plot_bars['precision'] + to_plot_bars['recall'])
                for offset, _k in enumerate(to_plot_bars.keys()):
                    lns.append(cplt.bar(_mid_bins + 0.15 * (offset-1) * _width, to_plot_bars[_k], color=colors[offset], width=0.1*_width, label=_k, alpha=0.7))

                labels = [_k for _k in to_plot_bars.keys()]

            #cplt.legend()
            if xi==0:
                cplt.legend(lns, labels)
                cplt.set_title(method, fontsize=font)
            if feature == "Mvir":
                cplt.axvline(2e14, color='black')
            if col_i == 0:  cplt.set_ylabel(col_to_name[feature], fontsize=font)
            if _need_log(feature): cplt.set_xscale("log")
    return


def plot_all_hists_by_features_stack_components_overlay(df, thresholds, save_path=None, font=DEFAULT_FONT, nbins=6):
    print("Where the bars sit is the middle point of the interval")

    df = df.reindex()
    df['Mvir'] = df['Mvir'].map(np.log) / np.log(10)
    df['tSZ'] = df['tSZ'].map(np.log) / np.log(10)

    #bins = {"Mvir": np.linspace}
    #nrows = len(thresholds)
    nrows = 1
    col_to_name = {"rvir": "virial radius", "Mvir": "log_10(virial mass)", "redshift": "redshift", "tSZ": "log_10(tSZ)"}
    print("From left to right, we have {}".format(", ".join(list(col_to_name.values()))))
    print("In each plot we have predicted-positives by {}".format(", ".join(thresholds.keys())))
    print("We are resticting the population to all positive cutouts")
    assert all([df[list(col_to_name.keys())[i]].count() == df[list(col_to_name.keys())[i - 1]].count() for i in
                range(1, len(col_to_name))])
    assert all([c in df.columns for c in col_to_name.keys()])
    ncols = len(col_to_name)
    #f = plt.figure(figsize=(4 * ro, 4 * nc))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))

    for xi, feature in enumerate(col_to_name.keys()):
        cplt = axes[xi]
        bins = None
        colors = CB_color_cycle.copy()
        for k in thresholds.keys():
            #kwargs = {"histtype":"step", "fill":True, "alpha": 0.3, 'color': colors.pop()}
            kwargs = {"histtype": "stepfilled", "alpha":0.3,  'color': colors.pop()}
            if bins is None:
                _, bins, _ = cplt.hist(df[df['pred_%s'%k] > thresholds[k]][feature], nbins, label=k, **kwargs)
            else:
                cplt.hist(df[df['pred_%s' % k] > thresholds[k]][feature], label=k, bins=bins, **kwargs)
        #_, bins, _ = cplt.hist([df[df['pred_%s'%k] > thresholds[k]][feature] for k in thresholds.keys()], nbins, label=list(thresholds.keys()), color=CB_color_cycle.copy(), stacked=True)
        #print("Bins:{}".format(bins))

        cplt.legend()
        cplt.set_title(col_to_name[feature], fontsize=font)
    return


#============================hardcoded things
def make_full_validtestdf(add_prediction=True, one_to_many=False, calibrate=False):
    df = pd.read_pickle(CACHE_FULLDF if not calibrate else CACHE_FULLDF_DIST2EDGE_CAL)
    df = df.drop(['halo_id', 'redshift', 'tSZ', 'Mvir', 'rvir', 'CL_ra', 'CL_dec'], axis=1)
    halos = pd.read_pickle(CACHE_MAPPED_HALOS).reset_index().reindex(columns=['halo_id', 'redshift', 'tSZ', 'Mvir', 'rvir', 'CL_ra', 'CL_dec', 'cutout_id'])
    halos = halos.sort_values('Mvir', ascending=False).drop_duplicates(subset=['cutout_id'], keep='first')
    df = df.merge(halos, how='outer' if one_to_many else 'left', left_on='cutout_id', right_on='cutout_id')
    df = df.rename(columns={'map_%s' % c: "cutout_%s" % c for c in ['ra', 'dec']})
    df['pred_MF'] = df['StoN']

    if calibrate:
        CNN_thres, MF_thres, product_thres = 0.36624489795918364, 13.408163265306122, 0.938469387755102
        CNN_AND_thres, MF_AND_thres =(0.2571428571428572, 7.938775510204081)
    else:
        CNN_thres, MF_thres, product_thres = 0.6959183673469387, 13.408163265306122, 0.9442857142857143
        CNN_AND_thres, MF_AND_thres = 0.6428571428571428, 6.591836734693878

    if add_prediction:
        #df = df.rename(columns={'pred':"CNN"})
        df['StoN'] = df['StoN'].fillna(2.)
        df['CNN_pred'] = df['pred_skymap'] > CNN_thres
        df['MF_pred'] = df['StoN'] > MF_thres
        df['EnsembleAND_pred'] = (df['pred_skymap'] > CNN_AND_thres) & (df['StoN'] > MF_AND_thres)
        df['EnsemblePROD'] = np.NaN
        for which in ['test', 'valid']:
            tdf = df[df['which'] == which]
            cnn_pred = tdf['pred'].rank() / float(tdf['pred'].count())
            mf_pred = tdf['StoN'].rank() / float(tdf['StoN'].count())
            df['EnsemblePROD'].update(cnn_pred * mf_pred)
        df['EnsemblePROD_pred'] = df['EnsemblePROD'] > product_thres
    return df

def make_full_testdf(add_prediction=True, calibrate=True):
    df = make_full_validtestdf(add_prediction=add_prediction, calibrate=calibrate)
    return df[df['which'] == 'test']

import numpy as np

def calc_angular_size(redshift, r, cosmo = None):
    from astropy.cosmology import WMAP5
    cosmo = cosmo or WMAP5
    '''
    Use redshift and radius (in Mpc) to convert to angular size in arcmin.
    Defaults to WMAP5 cosmology.
    '''
    d_A = cosmo.angular_diameter_distance(redshift)
    # angles in stupid astropy units
    theta = r / d_A
    return np.rad2deg(theta.value) * 60


#================================

def run_experiment(df, label, MFxlim=(1., 20.), CNNxlim=(0.2, 0.8), title="M>=2e14 & z>=0.25"):
    n_pos = label.sum()
    n_pos_test = label[df['which'] == 'test'].sum()
    print("There are %d(valid) + %d(test) = %d such positves (out of %d)\n\n" % (n_pos-n_pos_test, n_pos_test, n_pos, len(df)))
    def _run_valid_and_test(tdf, label, col, name, xlim):
        vdf = tdf[tdf['which'] == 'valid']
        tdf = tdf[tdf['which'] == 'test']
        #thres, val_F1 = get_F1(vdf[col].fillna(2.), y.reindex(vdf.index), xlim=xlim, method=name)
        x, y = get_F1(vdf[col].fillna(2.), label.reindex(vdf.index), xlim=xlim, method=name, get_raw=True, plot=False)
        thres, val_F1 = x[np.argmax(y)], np.max(y)
        F1 = _get_Fbeta(label.reindex(tdf.index), tdf[col] > thres, debug=True)
        print("Val F1: {}, test F1: {}, threshold: {}\n----{}---\n \n".format(val_F1, F1, thres, name))
        return x, y, thres, val_F1, F1

    xmf, ymf, _, _, mfF1 = _run_valid_and_test(df, label, 'StoN', 'MF', MFxlim)
    xcnn, ycnn, _, _, cnnF1 = _run_valid_and_test(df, label, 'pred_skymap', 'CNN', CNNxlim)
    stack_F1(xmf, ymf, xcnn, ycnn, title=title)
    return mfF1, cnnF1

def compare_df(df1, df2):
    cols = df1.columns.intersection(df2.columns)
    tdf1 = df1.reindex(columns=cols)
    tdf2 = df2.reindex(columns=cols)
    bdf = pd.concat([tdf1, tdf2], axis=0, ignore_index=True)
    n1, n2 = len(tdf1), len(tdf2)
    diff1 = bdf.drop_duplicates(keep='first')
    pass