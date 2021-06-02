import os
CUR_DIR = os.path.dirname(os.path.realpath(__file__))

DATA_PATH = 'Z:/deepsz'

CNN_MODEL_OUTPUT_DIR = os.path.abspath(os.path.join(CUR_DIR, './CNN/deepsz1')) #'/media/zhen/Research/deepsz_pytorch_2'


FULL_DATA_PATH = os.path.join(DATA_PATH, 'maps/reso0.25_small')
FULL_DATA_LABEL_PATH = os.path.join(DATA_PATH, 'maps/reso0.25_small/z0.25_mvir2e+14_label.pkl')
VARYING_DIST_DATA_PATH = os.path.join(CUR_DIR, 'data/maps/varying_dist_to_center10x')

CACHING_DIR = os.path.abspath(os.path.join(CUR_DIR, "./data/cache"))
CACHE_MAPPED_HALOS = os.path.join(CACHING_DIR, 'halos_mapped_to_cutouts.pkl')
CACHE_FULLDF = os.path.join(CACHING_DIR, 'all_component_preds_w_MF.pkl')
CACHE_FULLDF_DIST2EDGE = os.path.join(CACHING_DIR, 'all_component_preds_w_MF_dist2edge.pkl')
CACHE_FULLDF_DIST2EDGE_CAL = os.path.join(CACHING_DIR, 'all_component_preds_w_MF_dist2edge_calibrated.pkl') #This is the calibrated CNN "probabilities"
CACHE_CNNFAILURE = os.path.join(CACHING_DIR, 'CNNFailures.pkl')

MF_OUTPUT_PATH = os.path.abspath(os.path.join(CACHING_DIR, "../10mJy-ptsrcs_catalog.npz.npz"))
