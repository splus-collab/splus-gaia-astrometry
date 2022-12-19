# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
from astropy.io import ascii, fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates import Angle
from astroquery.vizier import Vizier


def get_gaia(tile_coords):
    """query gaia DR3 for photometry products"""

    print('querying gaia/vizier')
    v = Vizier(columns=['*', 'RAJ2000', 'DEJ2000'], catalog='I/355')
    v.ROW_LIMIT = 999999999
    gaia_data = v.query_region(tile_coords, radius=Angle(1.0, "deg"))[0]
    mask = gaia_data['RAJ2000'].mask & gaia_data['DEJ2000'].mask
    gaia_data = gaia_data[~mask]
    print('gaia_data is', gaia_data)

    return gaia_data

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tiles = ascii.read('tiles_new_status.csv')
    tile = 'SPLUS-s32s40'
    sra, sdec = tiles['RA'][tiles['NAME'] == tile], tiles['DEC'][tiles['NAME'] == tile]
    tile_coords =SkyCoord(ra=sra[0], dec=sdec[0], unit=(u.hour, u.deg), frame='icrs', equinox='J2000')
    gaia_data = get_gaia(tile_coords)

    scat = fits.open('/storage2/share/MAR-gaia-astrometry/' + tile + '_R.detection.cat')[2].data
    splus_coords = SkyCoord(ra=scat['ALPHA_J2000'], dec=scat['DELTA_J2000'], unit=(u.deg, u.deg))
    gaia_coords = SkyCoord(ra=gaia_data['RAJ2000'], dec=gaia_data['DEJ2000'], unit=(u.deg, u.deg))
    idx, d2d, d3d = splus_coords.match_to_catalog_3d(gaia_coords)
    separation = d2d < 5.0 * u.arcsec

    sample = (scat['MAG_AUTO'] > 13) & (scat['MAG_AUTO'] < 19)
    sample &= scat['FLAGS'] == 0
    # sample &= scat['CLASS_STAR' > 0.95] # MAR cat nao tem CLASS_STAR

    finalscat = scat[separation & sample]
    finalgaia = gaia_data[idx][separation & sample]

    abspm = abs(finalgaia['pmRA']) + abs(finalgaia['pmDE'])
    # get masked values in gaia
    mx = np.ma.masked_invalid(abspm)
    lmt = np.percentile(abspm[~mx.mask], 95)
    mask = (abspm < lmt) & ~mx.mask
    # calculate splus - gaia declination
    dediff = 3600. * (finalscat['DELTA_J2000']*u.deg - finalgaia['DEJ2000'])[mask]
    # calculate splus - gaia ra
    radiff = np.cos(finalscat['DELTA_J2000'])[mask] * 3600. * finalscat['ALPHA_J2000'][mask]
    radiff -= np.cos(finalgaia['DEJ2000'])[mask] * 3600. * finalgaia['RAJ2000'][mask]

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
