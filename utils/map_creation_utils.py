
import healpy as hp
import numpy as np
import reproject

def radec2pix(nside, ra, dec):
    return hp.pixelfunc.ang2pix(nside,np.radians(-dec+90.),np.radians(ra))


def h2f(hmap,target_header,nsideout):
        pr,footprint = reproject.reproject_from_healpix(
        (hmap, 'C'), target_header, #shape_out=(nsideout,nsideout),
        order='nearest-neighbor', nested=False)
        return pr

def f2h(fmap,target_header,nsideout):
        h2,footprint = reproject.reproject_to_healpix(
        (fmap, target_header), 'C',
        nside=nsideout, order='nearest-neighbor', nested=False)
        return h2

def gen_header(npix, reso,RA, dec):
    #generate square WCS headers
    #RA/dec default to be at center of square
    #reso in arcminute
    header="""
NAXIS   =                    2 / Number of data axes
NAXIS1  =               %1i /
NAXIS2  =               %1i /
DATE    = '2013-07-22'         / Creation UTC (CCCC-MM-DD) date of FITS header
CRVAL1  =              %10.9f /
CRVAL2  =              %10.9f /
CRPIX1  =              %3.2f /
CRPIX2  =              %3.2f /
CD1_1   =             %10.9f /
CD2_2   =             %10.9f /
CD2_1   =              0.00000 /
CD1_2   =              0.00000 /                                      
CTYPE1  = 'RA---ZEA'           /                                      
CTYPE2  = 'DEC--ZEA'           /                                      
CUNIT1  = 'deg     '           /                                      
CUNIT2  = 'deg     '           /                                      
CUNIT2  = 'deg     '           /
COORDSYS= 'icrs    '            /                                     
"""%(npix,npix,RA,dec,npix/2.0, npix/2.0, -reso/60.0, reso/60.0)
    return header


#======================================================Fixed version

import astropy.io.fits as fits
def get_map_from_bigsky(allc, ra, dec, reso, npix_ra, npix_dec):
    #Return a 2d numpy array, where
    # 1. the first axis is dec, and dec is increasing in the index
    # 2. the second axis is ra, and ra is DECREASING in the index
    header="""
NAXIS   =                    2 / Number of data axes
NAXIS1  =               %1i /
NAXIS2  =               %1i /
DATE    = '2013-07-22'         / Creation UTC (CCCC-MM-DD) date of FITS header
CRVAL1  =              %10.9f /
CRVAL2  =              %10.9f /
CRPIX1  =              %3.2f /
CRPIX2  =              %3.2f /
CD1_1   =             %10.9f /
CD2_2   =             %10.9f /
CD2_1   =              0.00000 /
CD1_2   =              0.00000 /                                      
CTYPE1  = 'RA---ZEA'           /                                      
CTYPE2  = 'DEC--ZEA'           /                                      
CUNIT1  = 'deg     '           /                                      
CUNIT2  = 'deg     '           /                                      
CUNIT2  = 'deg     '           /
COORDSYS= 'icrs    '            /                                     
"""%(npix_ra,npix_dec,ra,dec,npix_ra/2.0, npix_dec/2.0, -reso/60.0, reso/60.0)
    target_headerm = fits.Header.fromstring(header, sep='\n')
    nside = 8192
    return h2f(allc, target_headerm, nside)

import numpy as np
from astropy.cosmology import WMAP5

def calc_angular_size(redshift, r, cosmo = WMAP5):
    '''
    Use redshift and radius (in Mpc) to convert to angular size in arcmin.
    Defaults to WMAP5 cosmology.
    '''
    d_A = cosmo.angular_diameter_distance(redshift)
    # angles in stupid astropy units
    theta = r / d_A
    return np.rad2deg(theta.value) * 60
