# start with something simple
# list of clusters are mass > 2e14 M_solar; z > 0.5

# 0 < RA < 90; 0 < dec < 90
#
import ipdb
try:
    import astropy.io.fits as fits
    import utils.map_creation_utils as kutils
    from importlib import reload
    reload(kutils)
except:
    pass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
try:
    from ProgressBar import ProgressBar
except:
    from utils.ProgressBar import ProgressBar
from global_settings import DATA_PATH, CACHE_CNNFAILURE, VARYING_DIST_DATA_PATH

CUR_DIR = os.path.dirname(os.path.realpath(__file__))

class HalosCounter:
    def __init__(self, size=40, step=30, data_path = DATA_PATH, seed=0):
        self.size = size
        self.step = step

        self.CUTOUT_SIZE = size / 60. #in degree
        self.STEP_SIZE = step / 60. #In degree
        self.centers = np.arange(self.CUTOUT_SIZE / 2, 90 - self.CUTOUT_SIZE / 2, self.STEP_SIZE)
        self.N_STEPS = len(self.centers)
        self.data_path = data_path

        self.seed = seed

    def _map_deg2idx(self, ra_or_dec):
        ret = np.round((ra_or_dec - self.CUTOUT_SIZE / 2.) / self.STEP_SIZE).astype(int)
        return min(max(ret, 0), len(self.centers) - 1)
        #return np.floor(ra_or_dec / self.STEP_SIZE - 0.5).astype(int)

    def _map_idx2deg(self, idx):
        return idx * self.STEP_SIZE + self.CUTOUT_SIZE / 2.

    def _get_rand_sample(self, nsamples):
        np.random.seed(self.seed)
        return np.random.choice(self.N_STEPS ** 2, self.N_STEPS ** 2 if nsamples is None else nsamples, replace=False)

    def get_halos(self, z_min=None, mvir_min=None):
        cache_path = os.path.join(self.data_path, "_cache", "complete_halos.pkl")
        if os.path.isfile(cache_path):
            df = pd.read_pickle(cache_path)
        else:
            xrad = np.genfromtxt(os.path.join(self.data_path, "raw/halo_sz.ascii"), dtype=None)
            cols = {"redshift": 0, "tSZ": 13, "Mvir": 10, "rvir": 12, "CL_ra": 1, "CL_dec": 2}
            df = pd.DataFrame()
            for k in cols.keys():
                df[k] = xrad[:, cols[k]]
            df.to_pickle(cache_path)
        if z_min is not None: df = df[df['redshift'] >= z_min]
        if mvir_min is not None: df = df[df['Mvir'] >= mvir_min]
        df.index.name = "halo_id"
        return df

    def get_cutout_info(self):
        all_maps_idx = pd.MultiIndex.from_tuples([(i, j) for i in range(self.N_STEPS) for j in range(self.N_STEPS)],
                                                 names=['map_ra_idx', 'map_dec_idx'])
        df = pd.DataFrame(index=all_maps_idx).reset_index()
        df.index.name = "cutout_id"
        for c in ['ra', 'dec']: df["map_%s"%c] = df["map_%s_idx"%c].map(self._map_idx2deg)
        return df

def _recache_one(path, headers, cache_path):
    if not os.path.isfile(cache_path):
        with open(path, 'r') as f:
            df = [l.split() for l in f]
        df = pd.DataFrame(df, columns=headers).astype(float)
        try:
            df['halo_id'] = df['halo_id'].astype(int)
        except:
            pass
        df.to_pickle(cache_path)
    return cache_path

def recache_IR(path = os.path.join(DATA_PATH, 'raw', 'IRBlastPop.dat')):
    headers = ['halo_id', 'ra', 'dec', 'redshift', 'flux_30', 'flux_90', 'flux_148', 'flux_219', 'flux_277', 'flux_350']
    cache_path = os.path.join(DATA_PATH, "_cache", "high_flux_IR_galaxies.pkl")
    return pd.read_pickle(_recache_one(path, headers, cache_path))

def recache_Radio(path = os.path.join(DATA_PATH, 'raw', 'radio.cat')):
    headers = ['ra', 'dec', 'redshift', 'f_1.4', 'f_30', 'f_90', 'f_148', 'f_219', 'f_277', 'f_350']
    cache_path = os.path.join(DATA_PATH, "_cache", "radio_galaxies.pkl")
    return pd.read_pickle(_recache_one(path, headers, cache_path))

def match_IR():
    cache_path = os.path.join(DATA_PATH, "_cache", "high_flux_IR_galaxies_with_info.pkl")
    if not os.path.isfile(cache_path):
        self = CutoutGen(use_big=False, MULTIPLE=None)
        objs = recache_IR().rename(columns={"ra":"obj_ra", "dec":"obj_dec", 'halo_id':'ir_id'})
        self.df = self.halocounter.get_cutout_info()
        self.df2 = self.df.reset_index().set_index(['map_ra_idx', 'map_dec_idx'])
        self.df['which'] = self.df.apply(self._get_cuotout_set, axis=1)

        objs['map_ra_idx'] = objs['obj_ra'].map(self.halocounter._map_deg2idx).astype(int)
        objs['map_dec_idx'] = objs['obj_dec'].map(self.halocounter._map_deg2idx).astype(int)
        objs['cutout_id'] = objs.apply(lambda r: self.df2.loc[(r['map_ra_idx'], r['map_dec_idx'])]['cutout_id'], axis=1)
        objs['map_ra'] = objs['map_ra_idx'].map(self.halocounter._map_idx2deg)
        objs['map_dec'] = objs['map_dec_idx'].map(self.halocounter._map_idx2deg)
        pd.to_pickle(objs, cache_path)
    return pd.read_pickle(cache_path)

def match_radio():
    cache_path = os.path.join(DATA_PATH, "_cache", "radio_galaxies_with_info.pkl")
    if not os.path.isfile(cache_path):
        self = CutoutGen(use_big=False, MULTIPLE=None)
        objs = recache_Radio().rename(columns={"ra":"obj_ra", "dec":"obj_dec"})
        self.df = self.halocounter.get_cutout_info()
        self.df2 = self.df.reset_index().set_index(['map_ra_idx', 'map_dec_idx'])
        self.df['which'] = self.df.apply(self._get_cuotout_set, axis=1)

        objs['map_ra_idx'] = objs['obj_ra'].map(self.halocounter._map_deg2idx).astype(int)
        objs['map_dec_idx'] = objs['obj_dec'].map(self.halocounter._map_deg2idx).astype(int)
        objs['cutout_id'] = objs.apply(lambda r: self.df2.loc[(r['map_ra_idx'], r['map_dec_idx'])]['cutout_id'], axis=1)
        objs['map_ra'] = objs['map_ra_idx'].map(self.halocounter._map_idx2deg)
        objs['map_dec'] = objs['map_dec_idx'].map(self.halocounter._map_idx2deg)
        pd.to_pickle(objs, cache_path)
    return pd.read_pickle(cache_path)
class Annotator:
    def __init__(self, CUTOUT_SIZE, RESOLUTION=0.5):
        #CUTOUT_SIZE in degree
        self.RESOLUTION = RESOLUTION
        self.CUTOUT_SIZE = CUTOUT_SIZE
        self.npix = np.ceil(self.CUTOUT_SIZE * 60 / self.RESOLUTION)
    def ra_to_x(self, ra, cutout_ra):
        return self.npix * ((cutout_ra - ra) / self.CUTOUT_SIZE + 0.5)
    def dec_to_y(self, dec, cutout_dec):
        return self.npix * ((dec - cutout_dec) / self.CUTOUT_SIZE + 0.5)
    def x_to_ra(self, x, cutout_ra):
        return (0.5 - x / float(self.npix)) * self.CUTOUT_SIZE + cutout_ra
    def y_to_dec(self, y, cutout_dec):
        return (y / float(self.npix) - 0.5) * self.CUTOUT_SIZE + cutout_dec

class CutoutGen:

    CFAC = {90: 5.526540e8, 148: 1.072480e9, 219: 1.318837e9}
    FREQS = [90, 148, 219]
    RESOLUTION = 0.25 #arcmin

    def __init__(self, use_big=False, data_path=DATA_PATH,
                 min_redshift=0.25, min_mvir=2e14, MULTIPLE=None,
                 remove_overlap_in_train=True, split_exp_id=2):
        if use_big:
            size = 40
            step = 30
        else:
            size = 8
            step = 6
        self.big = use_big
        self.MULTIPLE=MULTIPLE
        suffix = "" if self.big else "_small"

        self.halocounter = HalosCounter(size, step, data_path=data_path)
        self.data_path = data_path

        #self.df2['cutout_id'] = self.df2['cutout_id'].astype(int)
        self.min_redshift = min_redshift
        self.min_mvir = min_mvir
        self.npix = int(np.ceil(self.halocounter.CUTOUT_SIZE * 60 / CutoutGen.RESOLUTION))

        self.annotator = Annotator(self.halocounter.CUTOUT_SIZE, CutoutGen.RESOLUTION)

        self._thres_str = "z{:.2f}_mvir{:.0e}".format(self.min_redshift, self.min_mvir)
        self.cache_dir = os.path.join(self.data_path, "_cache",
                                      "{:.2f}{}".format(CutoutGen.RESOLUTION, suffix),
                                      self._thres_str)
        self.cache_dir = self.cache_dir.replace("+", "")
        if not os.path.isdir(self.cache_dir): os.makedirs(self.cache_dir)

        self.annotation_dir = self.cache_dir.replace("_cache", "maps/annotations")
        if not os.path.isdir(self.annotation_dir): os.makedirs(self.annotation_dir)

        self.map_dir = os.path.join(self.data_path, "maps", "reso{:.2f}{}".format(CutoutGen.RESOLUTION, suffix))
        if not os.path.isdir(self.map_dir): os.makedirs(self.map_dir)

        self.label_path = os.path.join(self.map_dir, "%s_label.pkl"%self._thres_str)

        self.split_exp_id = split_exp_id
        _cutout_info_cache_path = os.path.join(self.cache_dir, "_cutout_info_df_split%d.pkl"%split_exp_id)
        if os.path.isfile(_cutout_info_cache_path):
            self.df = pd.read_pickle(_cutout_info_cache_path)
            self.df2 = self.df.reset_index().set_index(['map_ra_idx', 'map_dec_idx'])
        else:
            self.df = self.halocounter.get_cutout_info()
            self.df2 = self.df.reset_index().set_index(['map_ra_idx', 'map_dec_idx'])
            self.df['which'] = self.df.apply(self._get_cuotout_set, axis=1)
            self.df['y'] = False

            halos = self.halocounter.get_halos(z_min=self.min_redshift, mvir_min=self.min_mvir).sort_values('Mvir', ascending=False)
            #ipdb.set_trace()
            _map_deg2idx2 = lambda x: np.round(
                (x - self.halocounter.CUTOUT_SIZE / 2) / self.halocounter.STEP_SIZE).astype(int)
            halos['map_ra_idx'] = halos['CL_ra'].map(self.halocounter._map_deg2idx).astype(int)#.clip(0, len(self.halocounter.centers) - 1)
            halos['map_dec_idx'] = halos['CL_dec'].map(self.halocounter._map_deg2idx).astype(int)#.clip(0, len(self.halocounter.centers) - 1)
            #halos['map_ra_idx'] = halos['map_ra_idx'].astype(int)
            #halos['map_dec_idx'] = halos['map_dec_idx'].astype(int)
            halos['cutout_id'] = halos.apply(lambda r: self.df2.loc[(r['map_ra_idx'], r['map_dec_idx'])]['cutout_id'],
                                             axis=1)

            halos = halos.reset_index().reindex(columns=['halo_id', 'redshift', 'tSZ', 'Mvir', 'rvir', 'CL_ra', 'CL_dec', 'cutout_id'])
            halos = halos.sort_values('Mvir', ascending=False).drop_duplicates('cutout_id', keep='first')
            self.df = self.df.reset_index().merge(halos, how='left',on='cutout_id').set_index('cutout_id', verify_integrity=True)
            self.df['y'] = self.df['halo_id'].map(lambda x: not pd.isnull(x))
            pd.to_pickle(self.df, _cutout_info_cache_path)


        if not self.big:
            if remove_overlap_in_train:
                import functools
                #got rid of overlap in training set
                pos_idx = self.df[self.df['y']].set_index(['map_ra_idx', 'map_dec_idx']).index
                #pos_pair = self.df.loc[pos_idx].reset_index().reindex(columns=['map_ra_idx', 'map_dec_idx'])
                bad_idx = [pos_idx.map(lambda x: (x[0]+o0,x[1]+o1)) for o0 in [-1,1] for o1 in [-1,1]]
                bad_idx = functools.reduce(lambda x, y: x.union(y), bad_idx)
                not_overlapped_idx = self.df2.index.difference(bad_idx)
                not_overlapped_idx = self.df2.loc[not_overlapped_idx]['cutout_id'].reset_index().set_index('cutout_id').index
                self.df['overlapped'] = True
                self.df.loc[not_overlapped_idx, 'overlapped'] = False
                self.df = self.df[~((self.df['which'] == 'train') & (self.df['overlapped'] & ~self.df['y']))]
            #reduce the # of cutouts

            old_idx = self.df[self.df['which']!='train'].index

            pos_idx = self.df[self.df['y']].index
            if MULTIPLE is not None:
                np.random.seed(MULTIPLE)
                neg_idx = self.df.index.difference(pos_idx)
                neg_idx = neg_idx[np.random.permutation(len(neg_idx))]
                tdf = pd.concat([self.df.loc[pos_idx], self.df.loc[neg_idx[:MULTIPLE * len(pos_idx)]]])
                self.df = pd.concat([tdf, self.df.reindex(old_idx.difference(tdf.index))]).sort_index()
            #self.df.index = pd.Index([i for i in range(len(self.df))], name=self.df.index.name)
            self.df2 = self.df.reset_index().set_index(['map_ra_idx', 'map_dec_idx'])

    def faster_assign_cutout(self, halos):
        _map_degidx2 = lambda x: np.round((x - self.halocounter.CUTOUT_SIZE / 2) / self.halocounter.STEP_SIZE).astype(int)
        halos['ra_idx'] = halos['CL_ra'].map(_map_degidx2).clip(0, len(self.halocounter.centers) - 1)
        halos['dec_idx'] = halos['CL_dec'].map(_map_degidx2).clip(0, len(self.halocounter.centers) - 1)
        halos = halos.reset_index()
        halos = halos.merge(self.df2.rename(columns={"halo_id":"halo_id_old"}), how='left', left_on=['ra_idx', 'dec_idx'], right_index=True)
        halos = halos.sort_values('Mvir', ascending=False)
        halos = halos.drop_duplicates(subset=['ra_idx', 'dec_idx'], keep='first')
        return halos

    def get_df_with_halo_info(self):
        return self.df.merge(self.halocounter.get_halos(), left_on='halo_id', right_index=True, how='left')

    @staticmethod
    def halo_angle_old(halo_info,scaledown=0.6):
        #halo_info should be a dict with at least the following keys: ra, dec, rvir, redshift
        #
        d_in_Mpc = 4220 * halo_info['redshift']
        radius_theta = halo_info['rvir'] / d_in_Mpc / np.pi * 180
        return radius_theta * scaledown

    @staticmethod
    def halo_angle(halo_info,scaledown=0.6):
        H0 = 73.8 #km/sec/Mpc
        c = 299792.458 #km/sec
        q0 = -(0.5 * 0.264 - 0.736) #0.5 * Omega_m - Omega_lambda

        z = halo_info['redshift']
        da_in_Mpc = c / (H0 * q0**2) * (z * q0 + (q0-1)* (np.sqrt(2 * q0 * z + 1.) - 1.))  / (1 + z)**2
        radius_theta = halo_info['rvir'] / da_in_Mpc / np.pi * 180
        return radius_theta * scaledown
    @staticmethod
    def _map_component(c):
        component_list = ['samples', 'ksz', 'ir_pts', 'rad_pts', 'dust']
        COMPONENT_MAP = {'samples': ["lensedcmb", "tsz"]}
        for i, cc in enumerate(component_list):
            if i == 0: continue
            COMPONENT_MAP[cc] = COMPONENT_MAP[component_list[i-1]] + [cc]
        #COMPONENT_MAP = {"ksz": "ksz", "ir_pts": "ir_pts", "samples": ["lensedcmb", "tsz"],
        #                 "rad_pts": "rad_pts", "dust": "dust", "full": "full_sz", "skymap": "skymap"}
        #ipdb.set_trace()
        return COMPONENT_MAP[c] if isinstance(COMPONENT_MAP[c], list) else [COMPONENT_MAP[c]]

    def find_cutout(self, halo):
        ra_idx = np.argmin(np.abs(halo['CL_ra'] - self.halocounter.centers))
        dec_idx = np.argmin(np.abs(halo['CL_dec'] - self.halocounter.centers))
        return self.df2.loc[(ra_idx, dec_idx)]

    def annotate(self, halo, r=None):
        if r is None: r = self.find_cutout(halo)
        #x and y and everything are in terms of total width / height

        #ra_to_x = lambda ra: (r['map_ra'] - ra) / self.halocounter.CUTOUT_SIZE + 0.5
        #dec_to_y = lambda dec: (dec - r['map_dec']) / self.halocounter.CUTOUT_SIZE + 0.5
        #x_to_ra = lambda x: (0.5 - x) * self.halocounter.CUTOUT_SIZE + r['map_ra']
        #y_to_dec = lambda y: (y-0.5) * self.halocounter.CUTOUT_SIZE + r['map_dec']


        npix = np.ceil(self.halocounter.CUTOUT_SIZE * 60 / CutoutGen.RESOLUTION)
        theta = CutoutGen.halo_angle(halo) # in degree
        theta = min(theta, 6. / 60.)
        #theta = 3 / npix * self.halocounter.CUTOUT_SIZE #TODO: after experiemnt, remove this hardcode
        x = self.annotator.ra_to_x(halo['CL_ra'], r['map_ra'])
        y = self.annotator.dec_to_y(halo['CL_dec'], r['map_dec'])

        w = h = (2 * theta) / self.halocounter.CUTOUT_SIZE * npix

        #draw a circle
        r = theta * npix
        segmentations = []
        for angle in np.arange(0., 2 * np.pi, 2 * np.pi / 10.):
            #since it's symmetric, we don't need to worry about axes
            segmentations.extend([x + np.cos(angle) * r, y + np.sin(angle) * r])

        x, y = x - w / 2., y - h / 2.
        return (x, y, w, h), segmentations

    def gen_cutouts(self, freq=90, component="skymap"):
        result_dir = os.path.join(self.map_dir, 'components', "%s_freq%03i"%(component, freq))
        if not os.path.isdir(result_dir): os.makedirs(result_dir)

        raw_path = os.path.join(self.data_path, 'raw', '%03i_%s_healpix.fits'%(freq, component))
        assert os.path.isfile(raw_path)

        # get the map; convert from Jy/steradians -> dT/T_cmb (divided by 1.072480e9)
        print('loading %i ghz map' % freq)
        allc, header = kutils.hp.read_map(raw_path, h=True)
        allc = allc / CutoutGen.CFAC[freq]

        # this options are locked in. Have to be same as the options
        # that provide the input list of indices
        cutoutsize = self.halocounter.CUTOUT_SIZE * 60  # arcmin
        pixsize = CutoutGen.RESOLUTION
        npix = np.ceil(cutoutsize / pixsize)

        for ii in ProgressBar(range(len(self.df.index))):
            idx =self.df.index[ii]
            curr_path = os.path.join(result_dir, "%d.npy"%idx)
            if os.path.isfile(curr_path): continue
            fall = kutils.get_map_from_bigsky(allc, self.df.loc[idx, "map_ra"], self.df.loc[idx, "map_dec"], pixsize, npix, npix)
            np.save(curr_path, fall)
        del allc

    def _gen_noise(self, idx, noise_lev=1.):
        np.random.seed(idx)
        cutoutsize = self.halocounter.CUTOUT_SIZE * 60  # arcmin
        pixsize = CutoutGen.RESOLUTION
        npix = int(np.ceil(cutoutsize / pixsize))
        noise_scale = noise_lev / (180. * 60. / np.pi * np.sqrt((CutoutGen.RESOLUTION / 60. * np.pi / 180.) ** 2.))
        noise = np.random.standard_normal((npix, npix, len(CutoutGen.FREQS))) * noise_scale
        noise[:, :, 0] *= 2.8
        noise[:, :, 1] *= 2.6
        noise[:, :, 2] *= 6.6
        return noise

    def gen_plain_multifreq(self, component="skymap", wnoise=False):
        #This function is not used currently. It was used to only generate skymap
        scale = 1.0e6 * 2.726 #k2uk * Tcmb
        freqs = CutoutGen.FREQS
        result_dir = os.path.join(self.map_dir, 'components', "%s" % (component))
        result_dir_wnoise = os.path.join(self.map_dir, 'components', "%s(with noise)" % (component))
        if wnoise:
            for ii in ProgressBar(range(len(self.df.index))):
                idx = self.df.index[ii]
                curr_path = os.path.join(result_dir_wnoise, '%d.npy'%idx)
                if os.path.isfile(curr_path):
                    ipdb.set_trace()
                    continue
                curr_map = np.load(os.path.join(result_dir, '%d.npy'%idx)) + self._gen_noise(idx)
                np.save(curr_path, curr_map)
            return

        if not os.path.isdir(result_dir): os.makedirs(result_dir)
        if component == 'skymap':
            result_dirs = {f: os.path.join('/media/zhen/Data/deepsz/maps/reso0.25_small', "%s_freq%03i" % (component, f)) for f in freqs}
        else:
            result_dirs = {f: os.path.join(self.map_dir, 'components', "%s_freq%03i" % (component, f)) for f in freqs}
        for ii in ProgressBar(range(len(self.df.index))):
            idx = self.df.index[ii]
            curr_path = os.path.join(result_dir, '%d.npy'%idx)
            if os.path.isfile(curr_path): continue
            #ipdb.set_trace()
            curr_map = np.stack([np.load(os.path.join(result_dirs[f], "%d.npy"%idx)) for f in freqs], axis=2) * scale
            np.save(curr_path, curr_map)
        if component == 'skymap':
            return
        import shutil
        for dd in result_dirs.values(): shutil.rmtree(dd)

    def gen_multifreq_maps(self, component="skymap", with_noise=True):
        sub_components = self._map_component(component)
        if isinstance(sub_components, str): sub_components = [sub_components]
        result_dir = os.path.join(self.map_dir, "%s%s" % (component, "(with noise)" if with_noise else ""))
        component_dirs = {c: os.path.join(self.map_dir, 'components', c) for c in sub_components}
        if not os.path.isdir(result_dir): os.makedirs(result_dir)
        for ii in ProgressBar(range(len(self.df.index))):
            idx = self.df.index[ii]
            curr_path = os.path.join(result_dir, '%d.npy'%idx)
            if os.path.isfile(curr_path): continue
            curr_maps = [np.load(os.path.join(component_dirs[c], "%d.npy"%idx)) for c in sub_components]
            curr_map = sum(curr_maps) + (self._gen_noise(idx) if with_noise else 0.)
            np.save(curr_path, curr_map)

    def gen_labels(self):
        pd.to_pickle(self.df.reindex(columns=['which', 'y']), self.label_path)
        #np.save(self.label_path, self.df['y'].values)

    def plain_routine(self, component='skymap'):
        for freq in CutoutGen.FREQS:
            self.gen_cutouts(freq=freq, component=component)
        self.gen_plain_multifreq(component)


    def routine(self, component='samples'):
        #components = ['skymap', 'tsz', 'lensedcmb']
        subcomponents = self._map_component(component)
        for _comp in subcomponents:
            for freq in CutoutGen.FREQS:
                self.gen_cutouts(freq=freq, component=_comp)
        self.gen_multifreq(component)

    @staticmethod
    def _reorder_img_axes(arr):
        return np.swapaxes(np.swapaxes(arr, 0, 1), 1, 2)


    def _get_cuotout_set(self, r):
        if self.split_exp_id==2:
            if r['map_ra'] < 0.13 * 90: return 'test'
            if r['map_ra'] >= 0.2 * 90: return 'train'
            return 'valid'
        else:
            assert self.split_exp_id==1
            if r['map_ra'] > 0.75 * 90: return 'test'
            if r['map_ra'] <= 0.65 * 90: return 'train'
            return 'valid'

    def _get_info(self,which):
        return {"description": "%s set for the deepsz project"%which}
    def _get_license(self):
        return {"id":0, "name":"Nothing"}
    def _get_categories(self):
        return [{"supercategory":"object", "id":1, "name":"halo"}]
    def _get_annotations(self, which):
        assert which in ['train' ,'test', 'valid']
        annotations = []
        df = self.halocounter.get_halos(z_min=self.min_redshift, mvir_min=self.min_mvir)
        print(len(df))
        for i, idx in enumerate(df.index):
            cutout = self.find_cutout(df.loc[idx])
            if self._get_cuotout_set(cutout) != which: continue
            bbox, segmentations = self.annotate(df.loc[idx], cutout)
            curr_ = {"bbox": list(bbox),
                     "id": i,
                     "area": bbox[2] * bbox[3],
                     "segmentation": [segmentations],
                     "category_id": 1,
                     "image_id": int(cutout['cutout_id']),
                     'iscrowd': 0}
            annotations.append(curr_)
        return annotations
    def _get_images(self, which):
        images = []
        for i, idx in enumerate(self.df.index):
            if self._get_cuotout_set(self.df.loc[idx]) != which: continue
            images.append({"file_name":"%d.npy"%idx,
                           "id":idx,
                           "height": self.npix,
                           "width": self.npix,
                           "license": 0})
        return images
    def get_labels(self, which ='train'):
        import json
        fname = "labels_{}.json".format(which)
        cache_path = os.path.join(self.annotation_dir, fname)
        if not os.path.isfile(cache_path):
            res = {"info": self._get_info(which),
                   "licenses": self._get_license(),
                   "images": self._get_images(which),
                   "annotations": self._get_annotations(which),
                   "categories": self._get_categories()}
            json.dump(res, open(cache_path, 'w'))
        return json.load(open(cache_path, 'r'))

    def get_cutout(self, cutout_id, component='skymap',read=False, withnoise=True):
        if withnoise: component = component + "(with noise)"
        path = os.path.join(self.map_dir, component, '%d.npy'%cutout_id)
        if read: return np.load(path)
        return path

    def show_annotation(self, i=0, component='skymap'):
        res = self.get_labels()
        annotation = res['annotations'][i]
        img = self.get_cutout(annotation['image_id'], read=True)

        #img = CutoutGen._reorder_img_axes(img)
        x,y,w,h = annotation['bbox']
        #print(x,y,w,h)
        x1, x2 = int(round(x)), int(round(x + w))
        y1, y2 = int(round(y)), int(round(y + h))
        #print(x1, x2, y1, y2)
        x1, y1 = max(x1,0),max(y1,0)
        x2, y2 = min(x2, self.npix-1), min(y2, self.npix-1)
        #print(x1,x2,y1,y2)
        highlight_value = img.max()
        img[y1:y2+1, x1, :] = highlight_value
        img[y1:y2+1, x2, :] = highlight_value
        img[y1, x1:x2+1, :] = highlight_value
        img[y2, x1:x2+1, :] = highlight_value

        for i in range(3):
            plt.subplot(1,3,i+1)
            plt.imshow(img[:,:,i])
        return img, annotation

    def _read_detections(self, path=None, exp=None):
        import json
        if path is None:
            path = os.path.join(self.data_path, 'detectron_tmp_output')
        if exp is not None:
            assert isinstance(exp, str)
            path = os.path.join(path, exp, "test","deepsz1_test","generalized_rcnn")

        detections = json.load(open(os.path.join(path, "bbox_deepsz1_test_results.json"), 'r'))
        df = []
        for d in detections:
            df.append({"image_id":d['image_id'], 'score':d['score'],
                       'bbox_x':d['bbox'][0], 'bbox_y':d['bbox'][1],
                       'bbox_w':d['bbox'][2], 'bbox_h':d['bbox'][3],})
        df = pd.DataFrame(df).reindex(columns=['image_id', 'score', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'])

        truth = self.get_labels('test')

        dftruth = []
        for d in truth['annotations']:
            dftruth.append({"image_id":d['image_id'],
                       'bbox_x':d['bbox'][0], 'bbox_y':d['bbox'][1],
                       'bbox_w':d['bbox'][2], 'bbox_h':d['bbox'][3],})
        dftruth = pd.DataFrame(dftruth).reindex(columns=['image_id', 'score', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'])


        return df, dftruth, truth, path

    def get_detections(self, exp=None, NMS=None):
        dfp, dft, _, path= self._read_detections(exp=exp)

        if NMS is not None:
            NMS_cache_path = os.path.join(path, 'NMS_{}.pkl'.format(NMS))
            if not os.path.isfile(NMS_cache_path):
                all_dfs = []
                for img_id in dfp['image_id'].unique():
                    _dfp = dfp[dfp['image_id'] == img_id]
                    boxes = np.stack([_dfp['bbox_x'], _dfp['bbox_y'], _dfp['bbox_x']+_dfp['bbox_w'], _dfp['bbox_y']+_dfp['bbox_h']], axis=1)
                    if NMS == 'score':
                        _dfp = _dfp.iloc[NMS_by_score(boxes, _dfp['score'].values)]
                    else:
                        _dfp = _dfp.iloc[NMS_fast(boxes)]
                    all_dfs.append(_dfp)
                dfp = pd.concat(all_dfs)
                pd.to_pickle(dfp, NMS_cache_path)

            dfp = pd.read_pickle(NMS_cache_path)


        df = pd.concat([dfp,dft])
        df['cutout_ra'] = df['image_id'].map(lambda x: self.df.loc[x]['map_ra'])
        df['cutout_dec'] = df['image_id'].map(lambda x: self.df.loc[x]['map_dec'])
        df['bbox_ra'] = df.apply(lambda r: self.annotator.x_to_ra(r['bbox_x'] + 0.5 * r['bbox_w'], r['cutout_ra']),
                                 axis=1)
        df['bbox_dec'] = df.apply(lambda r: self.annotator.y_to_dec(r['bbox_y'] + 0.5 * r['bbox_h'], r['cutout_dec']),
                                 axis=1)
        df['ispred'] = df['score'].map(lambda x: not pd.isnull(x))
        return df


    def _get_detections_check(self):
        annotations = []
        df = self.halocounter.get_halos(z_min=self.min_redshift, mvir_min=self.min_mvir)
        ndf = []
        for i, idx in enumerate(df.index):
            cutout = self.find_cutout(df.loc[idx])
            if self._get_cuotout_set(cutout) != 'test': continue
            ndf.append({"cutout_ra": cutout['map_ra'], 'cutout_dec': cutout['map_dec'],
                        'image_id': cutout['cutout_id'],
                        'bbox_ra':df.loc[idx,'CL_ra'], 'bbox_dec':df.loc[idx,'CL_dec']})
        return pd.DataFrame(ndf)

    def show_test(self, i=0, thres=0.9, nms=None, exp='exp1', print_info=False):
        #detections, truth = self._read_detections()
        if print_info:
            __tdf = self.get_detections(exp=exp)
            pdf, tdf = __tdf[__tdf['ispred']], __tdf[~__tdf['ispred']]
        else:
            pdf, tdf, _ = self._read_detections(exp=exp)
        #cutout_id = truth['images'][i]['id']
        pdf = pdf[pdf['score'] > thres]
        if nms is not None:
            boxes = np.stack([pdf['bbox_x'], pdf['bbox_y'], pdf['bbox_x'] + pdf['bbox_w'], pdf['bbox_y'] + pdf['bbox_h']], axis=1)
            if nms == 'score':
                pdf = pdf.iloc[NMS_by_score(boxes, pdf['score'].values)]
            else:
                pdf = pdf.iloc[NMS_fast(boxes)]
        cutout_id = pdf['image_id'].unique()[i]
        #cutout_id = 31738
        tdf = tdf[tdf['image_id'] == cutout_id]
        pdf = pdf[pdf['image_id'] == cutout_id]
        if print_info:
            for i, row in pdf.iterrows():
                pass
            print(pdf.reindex(columns=['bbox_ra', 'bbox_dec', 'cutout_ra', 'cutout_dec', 'image_id', 'score', '']))

        img = self.get_cutout(cutout_id, read=True)
        img2 = img.copy()
        vtrue, vdetect = np.max(img), np.min(img)
        print(vtrue, vdetect)
        print("%d true boxes" % len(tdf))
        for _, x in tdf.iterrows():
            img = _draw_bbox(img, [x['bbox_x'],x['bbox_y'],x['bbox_w'],x['bbox_h']], vtrue)
        for _, x in pdf.iterrows():
            img2 = _draw_bbox(img2, [x['bbox_x'],x['bbox_y'],x['bbox_w'],x['bbox_h']], vdetect)

        for j in range(3):
            plt.subplot(2,3,j+1)
            plt.imshow(img[:,:,j])
            plt.subplot(2,3,j+1+ 3)
            plt.imshow(img2[:,:,j])
        return None

    def get_full_df(self, which='test'):
        cache_path = {K: os.path.join(self.cache_dir, "_full_df_%s.pkl"%K) for K in ['test', 'valid', 'train']}
        if not os.path.isfile(cache_path[which]):
            df = self.halocounter.get_halos(z_min=self.min_redshift, mvir_min=self.min_mvir)
            df['which'] = df.apply(lambda r: self._get_cuotout_set(self.find_cutout(r)), axis=1)
            df['halo_angle'] = df.apply(lambda r: CutoutGen.halo_angle(r, scaledown=1.0), axis=1)
            for k in cache_path.keys(): df[df['which'] == k].to_pickle(cache_path[k])
        return pd.read_pickle(cache_path[which])


class ShiftBadCNNGenCutout:
    """
    Class used to generate shifted CNN input cutouts
    """
    DIST_TO_RIM = 3. / 60
    RESOLUTION = 0.25
    def __init__(self, split=1, data_path=DATA_PATH, MULTIPLE=10,
                 cnn_failure_path=CACHE_CNNFAILURE,
                 map_output_dir=VARYING_DIST_DATA_PATH):
        self.split=split
        #self.failed_df = pd.read_pickle("../data/split_%d_CNNfailures.pkl"%split)
        self.failed_df = pd.read_pickle(cnn_failure_path or "../data/split_%d_CNNfailures.pkl"%split)
        self.halos = self.failed_df.drop_duplicates(['halo_id']).set_index('halo_id')
        self.MULTIPLE = MULTIPLE
        #self.map_dir = "../data/maps/split%d_%dx"%(split, MULTIPLE)
        self.map_dir = map_output_dir or "../data/maps/split%d_%dx"%(split, MULTIPLE)
        if not os.path.isdir(self.map_dir): os.makedirs(self.map_dir)
        self.halocounter = HalosCounter(8, 6, data_path=data_path)
        self.data_path = data_path
        self.radius_nsteps = 10
        self.df = self.gen_df().set_index('cutout_id_new')

    def gen_df(self):
        cutouts = []

        for halo_id, r in self.halos.iterrows():
            halo_id = int(halo_id)
            for step in range(self.radius_nsteps+1):
                ratio = float(step) / self.radius_nsteps
                for ii in range(self.MULTIPLE):
                    if ii != 0 and step == 0: continue
                    ra_offset, dec_offset = self.random_dist_offsets(ratio=ratio, seed=(halo_id * step) * self.MULTIPLE + ii)
                    #if step > 1: ipdb.set_trace()
                    cutouts.append({"halo_id":halo_id,
                                    "cutout_ra": r['CL_ra'] + ra_offset,
                                    "cutout_dec": r['CL_dec'] + dec_offset,
                                    "ratio": ratio})
        cutouts = pd.DataFrame(cutouts).reset_index().rename(columns={"index":"cutout_id_new"})
        cutouts = cutouts.merge(self.halos.reindex(columns=['redshift', 'Mvir', 'rvir', 'tSZ', 'y', 'CL_ra', 'CL_dec']),
                                left_on='halo_id', right_index=True)

        return cutouts

    @staticmethod
    def random_dist_offsets(ratio=0.5, seed=1):
        radius = ratio * ShiftBadCNNGenCutout.DIST_TO_RIM
        np.random.seed(seed)
        theta = np.random.rand() * 2 * np.pi
        ra, dec = np.cos(theta) * radius, np.sin(theta) * radius
        return ra, dec

    def _gen_noise(self, idx, noise_lev=1.):
        np.random.seed(idx)
        cutoutsize = self.halocounter.CUTOUT_SIZE * 60  # arcmin
        pixsize = CutoutGen.RESOLUTION
        npix = int(np.ceil(cutoutsize / pixsize))
        noise_scale = noise_lev / (180. * 60. / np.pi * np.sqrt((CutoutGen.RESOLUTION / 60. * np.pi / 180.) ** 2.))
        noise = np.random.standard_normal((npix, npix, len(CutoutGen.FREQS))) * noise_scale
        noise[:, :, 0] *= 2.8
        noise[:, :, 1] *= 2.6
        noise[:, :, 2] *= 6.6
        return noise

    def gen_cutout(self, freq=90):
        component='skymap'
        result_dir = os.path.join(self.map_dir, 'components', "%s_freq%03i"%(component, freq))
        if not os.path.isdir(result_dir): os.makedirs(result_dir)

        raw_path = os.path.join(self.data_path, 'raw', '%03i_%s_healpix.fits'%(freq, component))
        assert os.path.isfile(raw_path)

        # get the map; convert from Jy/steradians -> dT/T_cmb (divided by 1.072480e9)
        print('loading %i ghz map' % freq)
        allc, header = kutils.hp.read_map(raw_path, h=True)
        allc = allc / CutoutGen.CFAC[freq]

        # this options are locked in. Have to be same as the options
        # that provide the input list of indices
        cutoutsize = self.halocounter.CUTOUT_SIZE * 60  # arcmin
        pixsize = CutoutGen.RESOLUTION
        npix = np.ceil(cutoutsize / pixsize)

        for ii in ProgressBar(range(len(self.df.index))):
            idx =self.df.index[ii]
            curr_path = os.path.join(result_dir, "%d.npy"%idx)
            if os.path.isfile(curr_path): continue
            fall = kutils.get_map_from_bigsky(allc,
                                              self.df.loc[idx, "cutout_ra"],
                                              self.df.loc[idx, "cutout_dec"], pixsize, npix, npix)
            np.save(curr_path, fall)
        del allc

    def gen_plain_multifreq(self, component="skymap", wnoise=False):
        scale = 1.0e6 * 2.726 #k2uk * Tcmb
        freqs = CutoutGen.FREQS
        if wnoise:
            result_dir = os.path.join(self.map_dir, 'components', "%s(with noise)" % (component))
        else:
            result_dir = os.path.join(self.map_dir, 'components', "%s" % (component))

        if not os.path.isdir(result_dir): os.makedirs(result_dir)
        result_dirs = {f: os.path.join(self.map_dir, 'components', "%s_freq%03i" % (component, f)) for f in freqs}
        for ii in ProgressBar(range(len(self.df.index))):
            idx = self.df.index[ii]
            curr_path = os.path.join(result_dir, '%d.npy'%idx)
            if os.path.isfile(curr_path): continue
            curr_map = np.stack([np.load(os.path.join(result_dirs[f], "%d.npy"%idx)) for f in freqs], axis=2) * scale
            np.save(curr_path, curr_map + (self._gen_noise(idx) if wnoise else 0.))

    def gen_labels(self):
        self.df.drop(['CL_ra','CL_dec'],axis=1).to_pickle(os.path.join(self.map_dir, "labels.pkl"))

    def gen_routine(self):
        for freq in CutoutGen.FREQS: self.gen_cutout(freq)
        self.gen_plain_multifreq(wnoise=True)
        self.gen_labels()


def _draw_bbox(img, bbox, v):
    fw, fh = img.shape[:2]
    x, y, w, h = bbox
    # print(x,y,w,h)
    x1, x2 = max(int(round(x)), 0), min(int(round(x + w)), fw-1)
    y1, y2 = max(int(round(y)), 0), min(int(round(y + h)), fh-1)
    img[y1:y2 + 1, x1, :] = v
    img[y1:y2 + 1, x2, :] = v
    img[y1, x1:x2 + 1, :] = v
    img[y2, x1:x2 + 1, :] = v
    return img

# def gen_noise(nmaps=N_STEPS**2, suffix=""):
def gen_noise(overlap=False, suffix=""):
    # generate stampes of noise with differen noise levels
    # to be added to the 90, 150, 220 freq bands
    # for SPT-3G 1500 sq deg patch, it's [2.8,2.6,6.6]uK-arcmin
    # (from Aug 29th Slack channel proposal_forecast)
    #
    # also generate a future survey for [1,1,2] uK-arcmin

    nmaps = HalosCounter(overlap).N_STEPS ** 2

    # base dir
    bdir = DATA_PATH

    npix = 10
    pixsize = 0.5  # arcmin
    dx = pixsize / 60.0 * np.pi / 180.0

    # for nlev in [1.0, 2.6, 2.8, 2.0, 6.6]: # can just scale the 1uK-arcmin up
    nlev = 1.0

    noisemap = np.empty((nmaps, npix, npix), dtype=np.float64)
    random.seed(29)
    for i in range(nmaps):

        noisemap[i] = np.random.standard_normal((npix, npix)) * nlev / (180. * 60. / np.pi * np.sqrt(dx * dx))
        if i % 1000 == 0: print(i)

    np.save(bdir + 'deepsz_sandbox/withcmb/noise_%iuK-arcmin_90%s.npy' % (nlev, suffix), noisemap)
    random.seed(39)
    for i in range(nmaps):

        noisemap[i] = np.random.standard_normal((npix, npix)) * nlev / (180. * 60. / np.pi * np.sqrt(dx * dx))
        if i % 1000 == 0: print(i)

    np.save(bdir + 'deepsz_sandbox/withcmb/noise_%iuK-arcmin_150%s.npy' % (nlev, suffix), noisemap)

    random.seed(40)
    for i in range(nmaps):

        noisemap[i] = np.random.standard_normal((npix, npix)) * nlev / (180. * 60. / np.pi * np.sqrt(dx * dx))
        if i % 1000 == 0: print(i)

    np.save(bdir + 'deepsz_sandbox/withcmb/noise_%iuK-arcmin_220%s.npy' % (nlev, suffix), noisemap)


def filter_close_to(df, ra, dec):
    df2 = df[(df['map_ra'] < ra + 0.17) & (df['map_ra'] > ra - 0.17) & (df['map_dec'] < dec + 0.17) & (
                df['map_dec'] > dec - 0.17)]
    return df2.sort_values(['map_ra', 'map_dec'])


if __name__ == "__main__":
    o = CutoutGen(MULTIPLE=None, use_big=False, remove_overlap_in_train=False)
    #o.plain_routine('lensedcmb')
    #o.gen_plain_multifreq('lensedcmb')
    #o.gen_plain_multifreq('skymap')
    #o.gen_multifreq_maps('samples')
    #o.plain_routine('ir_pts')
    #o.plain_routine('rad_pts')
    o.plain_routine('ksz')
    o.gen_multifreq_maps('ksz')
    o.gen_multifreq_maps('ir_pts')
    o.gen_multifreq_maps('rad_pts')
    o.gen_multifreq_maps('dust')
    o.gen_multifreq_maps('skymap')

    o = CutoutGen(MULTIPLE=None, use_big=False, remove_overlap_in_train=True)
    o.gen_labels()


    #for freq in [90, 148, 219]:
    #    o.gen_cutouts(freq, 'ir_pts')
    #    o.gen_cutouts(freq, 'rad_pts')
    #    o.gen_cutouts(freq, 'dust')
    #    o.gen_cutouts(freq, 'full_sz')
    #    o.gen_cutouts(freq, 'ksz')
    #o.gen_plain_multifreq('ir_pts')
    #o.gen_plain_multifreq('rad_pts')
    #o.gen_plain_multifreq('dust')
    #o.gen_plain_multifreq('ksz')
    #o.routine()

    #o.gen_labels()
    #o.gen_multifreq()
    pass

    # gen_cutouts_new(90)
    # gen_cutouts_new(148)
    # gen_cutouts_new(219)
    # gen_noise(suffix="_overlap")

    # gen_components(90)
    # gen_components(148)
    # gen_components(219)



