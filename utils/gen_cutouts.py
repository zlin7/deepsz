# start with something simple
# list of clusters are mass > 2e14 M_solar; z > 0.5
#
# 0 < RA < 90; 0 < dec < 90
#
import ipdb
try:
    import healpy as hp
    import reproject
    import astropy.io.fits as fits
    import utils.map_creation_utils as kutils
except:
    pass
import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
import pandas as pd
import sys
import os
import pickle
import datetime

import random
CUR_DIR = os.path.dirname(os.path.realpath(__file__))

from global_settings import DATA_PATH, CACHE_CNNFAILURE, VARYING_DIST_DATA_PATH

class HalosCounter:
    def __init__(self, overlap=False):
        self.overlap = overlap
        self.CUTOUT_SIZE = 5. / 60
        self.STEP_SIZE = self.CUTOUT_SIZE / (2. if overlap else 1.)
        self.N_STEPS = int(90 / self.STEP_SIZE) - 1

    def _map_deg2idx(self, ra_or_dec):
        return np.floor(ra_or_dec / self.STEP_SIZE - 0.5).astype(int)

    def _map_idx2deg(self, idx):
        return idx * self.STEP_SIZE + self.CUTOUT_SIZE / 2.

    def _get_rand_sample(self, nsamples):
        np.random.seed(0)
        return np.random.choice(self.N_STEPS ** 2, self.N_STEPS ** 2 if nsamples is None else nsamples, replace=False)

    def cnt(self, df, cache_path, keep_max_col='Mvir', nsamples=None, convert_back=True):
        '''
        Count the number of halos with redshift > `z_min` and M_vir > `mvir_min`.
        If `return_halos` is False, returns the number of halos in each cutout.
        If `return_halos` is True, returns the number of halos in each cutout, and
        the ra, dec, redshift and M_vir of each halo above the given limits
        '''
        sample = self._get_rand_sample(nsamples)
        if os.path.exists(cache_path): return pd.read_pickle(cache_path).reindex(sample)
        df = df.reindex()
        for dim in ['ra', 'dec']:
            df.loc[:, "map_%s"%dim] = self._map_deg2idx(df.loc[:, 'CL_%s'%dim])
        all_maps_idx = pd.MultiIndex.from_tuples([(i, j) for i in range(self.N_STEPS) for j in range(self.N_STEPS)],
                                                 names=['map_ra', 'map_dec'])
        map_infos = df.sort_values([keep_max_col], ascending=False).drop_duplicates(
            subset=['map_ra', 'map_dec']).reset_index().rename(columns={"index": "idx"}).set_index(
            ['map_ra', 'map_dec'])
        map_infos.loc[:, 'has_object'] = True

        full = map_infos.reindex(all_maps_idx).reset_index()
        full.loc[:, 'has_object'] = full.loc[:, 'has_object'].fillna(value=False)
        if convert_back:
            for dim in ['ra', 'dec']:
                full.loc[:, "map_%s" % dim] = self._map_idx2deg(full.loc[:, 'map_%s' % dim])
        full.to_pickle(cache_path)
        return full.reindex(sample)

    def get_complete_df(self, nsamples=None, data_path=DATA_PATH):
        cache_path = os.path.join(data_path, "_complete_df_{}overlap.pkl".format("" if self.overlap else 'non'))
        if os.path.isfile(cache_path): return pd.read_pickle(cache_path)
        z_min = 0.25
        mvir_min = 5e13
        mvir_max = None
        result_path = os.path.join(data_path,
                                   "_count_halos_new_{}_{:.0e}_{}_centers_in_degs.pkl".format(z_min, mvir_min, mvir_max,
                                                                                              "_overlap" if self.overlap else ""))

        print(result_path)
        df = self.cnt(gen_halos_complete(z_min=z_min, mvir_min=mvir_min, mvir_max=mvir_max), result_path,
                 keep_max_col='Mvir', nsamples=nsamples)
        pdb.set_trace()
        df = df.rename(columns={"has_object": "Truth(>5e13)", "map_ra": "cutout_ra", "map_dec": "cutout_dec"})
        df['Truth(>2e14)'] = (df['Mvir'] > 2e14) & (df['redshift'] > 0.5)

        for k in ['dec', 'ra']: df['map_%s_idx' % k] = self._map_deg2idx(df["cutout_%s" % k])
        df = df.reset_index().set_index(['map_dec_idx', 'map_ra_idx'])
        for thres in ['5e13', '2e14']:
            import functools
            pdf = df[df['Truth(>%s)' % thres]]
            bad_idx = [pdf.index.map(lambda x: (x[0] + o0, x[1] + o1)) for o0 in [-1, 1] for o1 in [-1, 1]]
            bad_idx = functools.reduce(lambda x, y: x.union(y), bad_idx)
            neg_idx = df.index.difference(bad_idx).difference(pdf.index)
            df.loc[:, 'Empty(>%s)' % thres] = False
            df.loc[neg_idx, 'Empty(>%s)' % thres] = True
        ret = df.reset_index().drop(['map_dec_idx', 'map_ra_idx'], axis=1).set_index('index')
        ret.to_pickle(cache_path)
        return ret

def get_halos(z_min=0.25, mvir_min=5e13, mvir_max=None):
    halos = np.load(os.path.join(DATA_PATH, 'halos.npz'))
    halos = pd.DataFrame({k:halos[k] for k in halos.keys()})
    if mvir_max is None: mvir_max = halos['mvir'].dropna().max()
    good_halos = halos[(halos['mvir'] > mvir_min) & (halos['mvir'] < mvir_max) & (halos['z'] >= z_min)]
    return good_halos

def gen_halos_complete(z_min=0.25, mvir_min=5e13, mvir_max=None):
    cache_path = os.path.join(DATA_PATH, "deepsz_sandbox", "complete_halos.pkl".format(z_min, mvir_min))
    print(cache_path)
    if os.path.isfile(cache_path):
        df = pd.read_pickle(cache_path)
    else:
        xrad = np.genfromtxt(os.path.join(DATA_PATH, "seghal09_maps/halo_sz.ascii"),dtype=None)

        cols = {"redshift":0, "tSZ":13, "Mvir":10,"rvir":12,"CL_ra":1, "CL_dec":2}
        df = pd.DataFrame()
        for k in cols.keys():
            df[k] = xrad[:, cols[k]]
        df.to_pickle(cache_path)
    df = df[(df['redshift'] >= z_min) & (df['Mvir'] > mvir_min)]
    if mvir_max is not None: df = df[df['Mvir'] <= mvir_max]
    return df 

def get_rads(flux_min=5.0):
    global DATA_PATH
    xrad_cache_path = os.path.join(DATA_PATH, "_gen_cutout_all_rads")
    if os.path.exists(xrad_cache_path):
        rads = pd.read_pickle(xrad_cache_path)
    else:
        xrad = np.genfromtxt(DATA_PATH+"seghal09_maps/radio.cat",dtype=None)
        rads = pd.DataFrame({"ra":xrad[:,0], "dec":xrad[:,1], "flux":xrad[:,6]})
        rads.to_pickle(xrad_cache_path)
    #use only flux > 5mJy at 150 radio sources
    good_rads= rads[rads['flux'] > flux_min]
    return good_rads

def _read_until(path, nsamples, step=100000, npix=10, nfreq=1, base_path=None):
    if base_path is None:
        samples = np.empty((nsamples,nfreq,int(npix),int(npix)),dtype=np.float64)
    else:
        samples = np.load(base_path)
    finished = -1
    for i in range(nsamples):
        if (i + 1) % step == 0:
            path_t = path.replace(".npy", "%d.npy"%i)
            if os.path.isfile(path_t):
                samples[(i-i%step):(i+1)] = np.load(path_t)
                finished = i
            else:
                break
    return samples, finished


def gen_cutouts_new(freq=90, nsamples=None, overlap=False, bdir = DATA_PATH):
    o = HalosCounter(overlap)
    cfac = {90:5.526540e8, 148:1.072480e9,219:1.318837e9}
    result_path = bdir+'deepsz_sandbox/withcmb/1025_skymap_freq%03i%s.npy'%(freq, "_overlap" if overlap else "")
    if os.path.isfile(result_path): return result_path

    nside = 8192

    #get the map; convert from Jy/steradians -> dT/T_cmb (divided by 1.072480e9)
    print('loading %i ghz map'%freq)
    allc,header=hp.read_map(bdir+'seghal09_maps/%03ighz_sims/%03i_skymap_healpix.fits'%(freq,freq),h=True)
    allc = allc/cfac[freq]

    # this options are locked in. Have to be same as the options 
    # that provide the input list of indices
    cutoutsize = o.CUTOUT_SIZE * 60 #arcmin
    pixsize = 0.5 #arcmin
    npix = np.ceil(cutoutsize/pixsize)

    # load the list 
    print('loading list')
    df = o.get_complete_df(nsamples).rename(columns={"cutout_ra":"map_ra", "cutout_dec":"map_dec"})
    nsamples = len(df.index)
    nfreq=1

    samples, finished = _read_until(result_path, nsamples)

    for ii, cutout_id in enumerate(df.index):
        if ii <= finished: continue
        ra1, dec1 = df.loc[cutout_id, "map_ra"], df.loc[cutout_id, "map_dec"]
        header = kutils.gen_header(npix, pixsize, ra1, dec1)
        target_headerm = fits.Header.fromstring(header, sep='\n')
            
        fall  = kutils.h2f(allc,target_headerm,nside)
        samples[ii,0,:,:]=fall

        if ii % 1000 == 0: print(ii, datetime.datetime.now())
        if (ii+1) % 100000 == 0: np.save('%s%d.npy'%(result_path, ii),samples[(ii - ii%100000):(ii+1)])
    np.save(result_path,samples)

def filter_close_to(df, ra, dec):
    df2 = df[(df['map_ra'] < ra + 0.17)&(df['map_ra'] > ra - 0.17)&(df['map_dec'] < dec + 0.17)&(df['map_dec'] > dec - 0.17)]
    return df2.sort_values(['map_ra', 'map_dec'])

# quick visual check

if __name__ == "__main__":

    #gen_cutouts_new(90)
    #gen_cutouts_new(148)
    #gen_cutouts_new(219)
    #gen_noise(suffix="_overlap")

    #gen_components(90)
    #gen_components(148)
    #gen_components(219)
    pass
    #for freq in [90, 148, 219]:
        ##gen_components(freq)
        #gen_components(freq)


