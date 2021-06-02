##=================================== Make dataframes that we need

import numpy as np
import pandas as pd
import sys
import os, ipdb
sys.path.append(os.path.abspath(".."))

from global_settings import MF_OUTPUT_PATH, CACHING_DIR, CACHE_MAPPED_HALOS, CACHE_FULLDF, CACHE_FULLDF_DIST2EDGE, CACHE_CNNFAILURE, VARYING_DIST_DATA_PATH


import utils.deepsz_main as dutils
import utils.utils2 as utils

Obj = dutils.CutoutGen(use_big=False,MULTIPLE=None)

#change data_path to the location where you tore all the detectron results
#the scructure here for example should be Data/deepsz/detectron_tmp_output/{experiment_name}
PytorchResults = utils.PytorchResultReader()
#PytorchResultsSamples = utils.PytorchResultReader(exp_name="ratio1-20_convbody=R-50-C4_SGD_lr=0.005_wd=0.003_steps=1000-4000_comp=samples")
master_df = Obj.get_df_with_halo_info().reset_index().rename(columns={"map_%s"%s:'cutout_%s'%s for s in ['ra','dec']})
pos_df = master_df[master_df['y']]
neg_df = master_df[~master_df['y']]



if not os.path.isfile(CACHE_MAPPED_HALOS):
    Obj_temp = dutils.CutoutGen(use_big=False,MULTIPLE=None, remove_overlap_in_train=False)
    halos = Obj_temp.halocounter.get_halos(z_min=0, mvir_min=0).sort_values('Mvir', ascending=False)
    _map_deg2idx2 = lambda x: np.round((x - Obj_temp.halocounter.CUTOUT_SIZE/2) / Obj_temp.halocounter.STEP_SIZE).astype(int)
    halos['map_ra_idx'] = halos['CL_ra'].map(_map_deg2idx2).clip(0,len(Obj_temp.halocounter.centers) - 1).astype(int)
    halos['map_dec_idx'] = halos['CL_dec'].map(_map_deg2idx2).clip(0,len(Obj_temp.halocounter.centers) - 1).astype(int)
    halos['cutout_id'] = halos.apply(lambda r: Obj_temp.df2.loc[(r['map_ra_idx'], r['map_dec_idx'])]['cutout_id'], axis=1)
    pd.to_pickle(halos, CACHE_MAPPED_HALOS)

if not os.path.isfile(CACHE_FULLDF):
    comp_F1s = {}
    df = Obj.df.reindex()
    # ==================Add the predictions
    for comp in ['samples', 'ksz', 'ir_pts', 'rad_pts', 'dust', 'skymap']:
        print(comp)
        _P = utils.PytorchResultReader(
            exp_name="ratio1-20_convbody=R-50-C4_SGD_lr=0.005_wd=0.003_steps=1000-4000_comp=%s" % comp)
        cnn_result = _P.get_best()[0]
        val_F1_thres = cnn_result['stat']['F1_thres']
        df['pred_%s' % comp] = cnn_result['pred'].drop(['true', 'which'], axis=1).iloc[:, 0]
        if comp == 'samples':
            df = df.dropna(subset=['pred_%s' % comp]).reindex(cnn_result['pred'].index)
            print(df.shape)
        assert df['which'].equals(cnn_result['pred']['which'])
        if comp == 'samples':
            df = df.merge(Obj.halocounter.get_halos(), right_index=True, left_on='halo_id', how='left')
        # utils.get_F1(df[df['which'] == 'test']['pred_%s'%comp], df[df['which'] == 'test']['y'])
        comp_F1s[comp] = utils._get_Fbeta(df[df['which'] == 'test']['y'],
                                          df[df['which'] == 'test']['pred_%s' % comp] > val_F1_thres, debug=True)
    print(comp_F1s)
    df['pred'] = df['pred_skymap']

    # ===========Add MF
    cat_mf = np.load('../data/10mJy-ptsrcs_catalog.npz')  # MF output
    col_map = [("mf_peaksig", "sig"), ("mf_dec", "dec"), ("mf_ra", "ra")]
    mf_df = pd.DataFrame({k1: cat_mf[k2].byteswap().newbyteorder() for k1, k2 in col_map}).reindex(
        columns=[x[0] for x in col_map])
    print("min sig={}".format(np.min(mf_df['mf_peaksig'])))
    mfhalo_df = utils.match(mf_df, 1, cache_dir='../data/cache_wPtsrcs')

    # =======================
    cat_cnn = df.reindex().reset_index()
    _map_deg2idx = lambda x: np.argmin(np.abs(x - Obj.halocounter.centers))
    mfhalo_df['ra_cutout'] = mfhalo_df['ra_halo'].fillna(mfhalo_df['mf_ra']).map(_map_deg2idx)
    mfhalo_df['dec_cutout'] = mfhalo_df['dec_halo'].fillna(mfhalo_df['mf_dec']).map(_map_deg2idx)

    tmf_df = mfhalo_df.reindex(columns=['ra_cutout', 'dec_cutout', 'mf_ra', 'mf_dec', 'StoN'])
    merged_df = pd.merge(cat_cnn.drop(['overlapped'], axis=1),
                         # mfhalo_df.rename(columns={"Mvir":"mf_Mvir", "redshift":"mf_redshift"}).drop('halo_id',axis=1),
                         tmf_df[tmf_df['mf_ra'] <= 0.2 * 90],
                         how='left', left_on=['map_ra_idx', 'map_dec_idx'], right_on=['ra_cutout', 'dec_cutout'])
    merged_df.drop(columns=['ra_cutout', 'dec_cutout'], inplace=True)
    merged_df = merged_df.sort_values(['StoN'], ascending=True).drop_duplicates(subset=['map_ra', 'map_dec'],
                                                                                keep='last')
    # merged_df = merged_df.drop(['map_ra_idx', 'map_dec_idx'],axis=1)
    merged_df.to_pickle(CACHE_FULLDF)

#New codes from https://github.com/deepskies/deepsz/blob/master/Analysis/8(6)arcmin_match_newtrainingset.ipynb on Jul 25 2020
if not os.path.isfile(CACHE_FULLDF_DIST2EDGE):
    df = pd.read_pickle(CACHE_FULLDF).set_index('cutout_id')
    df['pos_dist2edge'] = df.apply(lambda r: min(0.05 - abs(r['CL_ra'] - r['map_ra']),
                                                 0.05 - abs(r['CL_dec'] - r['map_dec'])) if r['y'] else np.NaN, axis=1)
    ref_df = df.reset_index().set_index(['map_ra_idx', 'map_dec_idx'])
    def _get_neighbor_dist(r, offset, dim):
        if dim == 'ra':
            idx = (r['map_ra_idx'] + offset, r['map_dec_idx'])
        else:
            idx = (r['map_ra_idx'], r['map_dec_idx'] + offset)
        try:
            return ref_df.loc[idx, 'pos_dist2edge']
        except KeyError:
            return np.NaN

    df['neg_dist2edge'] = df.apply(lambda r:
                                   pd.Series([_get_neighbor_dist(r, d,dim) for d in [-1, 1] for dim in ['ra','dec']]).dropna().min(),
                                   axis=1)
    df.to_pickle(CACHE_FULLDF_DIST2EDGE)


#CNN failures for second round CNN prediction (by manimuplating the distance from center)
if not os.path.isfile(CACHE_CNNFAILURE):
    df = pd.read_pickle(CACHE_FULLDF).rename(columns={'map_dec':'cutout_dec', 'map_ra':'cutout_ra'})
    for ss in ['dec', 'ra']: df['rel_%s'%ss] = df['CL_%s'%ss] - df['cutout_%s'%ss]
    df = df.reindex(columns=['cutout_id', 'cutout_ra', 'cutout_dec', 'which', 'y', 'halo_id', 'pred',
                             'redshift', 'tSZ', 'Mvir', 'rvir', 'CL_ra', 'CL_dec', 'mf_ra', 'mf_dec',
                             'StoN', 'rel_ra', 'rel_dec'])
    df = df[(df['StoN'] > 15.4) & (df['pred'] < 0.5) & (df['which'] == 'test') & (df['y'])]
    df.to_pickle(CACHE_CNNFAILURE)



#Shift the cutouts with different distance from center to evaluate edge effect
if not os.path.isfile(os.path.join(VARYING_DIST_DATA_PATH, 'labels.pkl')):
    dutils.ShiftBadCNNGenCutout().gen_routine()


