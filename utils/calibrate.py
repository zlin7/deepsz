import numpy as np
import pickle
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath("../utils"))
sys.path.append(os.path.abspath(".."))
import global_settings as sgs

import ipdb
import torch
import torch.nn as nn
class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15, is_prob=False):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.is_prob = is_prob

    def forward(self, logits, labels):
        if self.is_prob:
            confidences = logits
            accuracies = (logits > 0.5).eq(labels)
        else:
            softmaxes = torch.softmax(logits, dim=1)
            confidences, predictions = torch.max(softmaxes, 1)
            accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

class TemperatureScaling_Pytorch:
    def __init__(self, scores, labels,
                 lr=0.01, max_iter=50):
        self.scores = torch.tensor(scores)
        self.labels = labels
        self.temp = nn.Parameter(torch.ones(1) * 1.)

        pre_ece = _ECELoss()(scores, labels).item()

        optimizer = torch.optim.LBFGS([self.temp], lr=lr, max_iter=max_iter)
        def eval_func():
            l = self.scores / self.temp
            loss = torch.nn.CrossEntropyLoss(l, self.labels)
            loss.backward()
            return loss
        optimizer.step(eval_func)

        post_ece = _ECELoss()(self.transform(self.scores, to_prob=False), labels).item()

        print(f"{pre_ece}->{post_ece}")

    @classmethod
    def prob2logit(cls, p):
        return np.log(p / (1.-p))

    @classmethod
    def logit2prob(cls, l):
        return 1./(1.+np.exp(-l))

    def transform(self, scores, to_prob=False):
        l = scores / self.temp.items()
        if to_prob: l = self.logit2prob(l)
        return l


#=====================================================================================================================
#https://github.com/ethen8181/machine-learning/blob/master/model_selection/prob_calibration/calibration_module/calibrator.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
#from calibration_module.utils import create_binned_data, get_bin_boundaries

#__all__ = [
#    'HistogramCalibrator',
#    'PlattCalibrator',
#    'PlattHistogramCalibrator'
#]

class PlattCalibrator(BaseEstimator):
    """
    Boils down to applying a Logistic Regression.
    Parameters
    ----------
    log_odds : bool, default True
        Logistic Regression assumes a linear relationship between its input
        and the log-odds of the class probabilities. Converting the probability
        to log-odds scale typically improves performance.
    Attributes
    ----------
    coef_ : ndarray of shape (1,)
        Binary logistic regression's coefficient.
    intercept_ : ndarray of shape (1,)
        Binary logistic regression's intercept.
    """

    def __init__(self, log_odds: bool = True):
        self.log_odds = log_odds

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray):
        """
        Learns the logistic regression weights.
        Parameters
        ----------
        y_prob : 1d ndarray
            Raw probability/score of the positive class.
        y_true : 1d ndarray
            Binary true targets.
        Returns
        -------
        self
        """
        self.fit_predict(y_prob, y_true)
        return self

    @staticmethod
    def _convert_to_log_odds(y_prob: np.ndarray) -> np.ndarray:
        eps = 1e-12
        y_prob = np.clip(y_prob, eps, 1 - eps)
        y_prob = np.log(y_prob / (1 - y_prob))
        return y_prob

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Predicts the calibrated probability.
        Parameters
        ----------
        y_prob : 1d ndarray
            Raw probability/score of the positive class.
        Returns
        -------
        y_calibrated_prob : 1d ndarray
            Calibrated probability.
        """
        if self.log_odds:
            y_prob = self._convert_to_log_odds(y_prob)

        output = self._transform(y_prob)
        return output

    def _transform(self, y_prob: np.ndarray) -> np.ndarray:
        output = y_prob * self.coef_[0] + self.intercept_
        output = 1 / (1 + np.exp(-output))
        return output

    def fit_predict(self, y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Chain the .fit and .predict step together.
        Parameters
        ----------
        y_prob : 1d ndarray
            Raw probability/score of the positive class.
        y_true : 1d ndarray
            Binary true targets.
        Returns
        -------
        y_calibrated_prob : 1d ndarray
            Calibrated probability.
        """
        if self.log_odds:
            y_prob = self._convert_to_log_odds(y_prob)

        # the class expects 2d ndarray as input features
        logistic = LogisticRegression(C=1e10, solver='lbfgs')
        logistic.fit(y_prob.reshape(-1, 1), y_true)
        self.coef_ = logistic.coef_[0]
        self.intercept_ = logistic.intercept_

        y_calibrated_prob = self._transform(y_prob)
        return y_calibrated_prob

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
class Calibrate:
    def __init__(self, scores, labels,  method='platt'):
        assert method in {'iso', 'platt'}
        self.method = method
        if method == 'iso':
            pass
        else:
            self.obj = PlattCalibrator(True)
            self.obj.fit(scores, labels)
        pass

    @classmethod
    def prob2logit(cls, p):
        return np.log(p / (1.-p))

    @classmethod
    def logit2prob(cls, l):
        return 1./(1.+np.exp(-l))

    def pred(self, scores):
        if self.method == 'platt':
            return self.obj.predict(scores)

def cal_df(cols = ['pred'+s for s in ['', '_samples', '_ksz', '_ir_pts', '_rad_pts', '_dust', '_skymap']]):

    df = pd.read_pickle(sgs.CACHE_FULLDF_DIST2EDGE)  # Test val
    if not os.path.isfile(sgs.CACHE_FULLDF_DIST2EDGE_CAL):
        vdf = df[df['which']=='valid']
        tdf = df[df['which']=='test']
        new_df = df.copy()
        for col in cols:
            new_df[col] = np.NaN
            pre_ece = _ECELoss(is_prob=True)(torch.tensor(tdf[col].values), torch.tensor(tdf['y'].values)).item()
            print(col)
            print("Pre:", pre_ece)
            o = Calibrate(vdf[col].values, vdf['y'])

            new_df.loc[tdf.index, col] = o.pred(tdf[col].values)
            new_df.loc[vdf.index, col] = o.pred(vdf[col].values)
            post_ece = _ECELoss(is_prob=True)(torch.tensor(new_df.loc[tdf.index, col]), torch.tensor(tdf['y'].values)).item()
            print("Post:", post_ece)
        pd.to_pickle(new_df, sgs.CACHE_FULLDF_DIST2EDGE_CAL)
    return pd.read_pickle(sgs.CACHE_FULLDF_DIST2EDGE_CAL), df

if __name__ == '__main__':
    new_df, df = cal_df()
