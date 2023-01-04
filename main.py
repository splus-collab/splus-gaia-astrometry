# This module is meant to calculate the differences between the astrometry from S-PLUS to that
# of Gaia DR2 or DR3
# Herpich F. R. 2022-12-20 fabiorafaelh@gmail.com
# GitHub: herpichfr
# ORCID: 0000-0001-7907-7884

import os
import numpy as np
from astropy.io import ascii, fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates import Angle
from astroquery.vizier import Vizier
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import sys
import time
from pathlib import Path
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

def get_gaia(workdir, tile_coords, tilename, gaia_dr):
    """query gaia DR3 for photometry products"""

    print('querying gaia/vizier')
    gaia_dr = gaia_dr
    if gaia_dr == 'DR2':
        catalognumb = '345'
    elif gaia_dr == 'DR3':
        catalognumb = '355'
    else:
        raise IOError('Check Gaia DR number via vizier page')

    v = Vizier(columns=['*', 'RAJ2000', 'DEJ2000'], catalog='I/'+catalognumb)
    v.ROW_LIMIT = 999999999
    gaia_data = v.query_region(tile_coords, radius=Angle(1.0, "deg"))[0]
    mask = gaia_data['RAJ2000'].mask & gaia_data['DEJ2000'].mask
    gaia_data = gaia_data[~mask]
    print('gaia_data is', gaia_data)

    gaia_cat_path = workdir + 'gaia_' + gaia_dr + '/' + tilename + '_gaiacat.csv'
    if not os.path.isdir(workdir + 'gaia_' + gaia_dr):
        os.mkdir(workdir + 'gaia_' + gaia_dr)

    print('saving result of match with gaia to', gaia_cat_path)
    gaia_data.to_pandas().to_csv(gaia_cat_path, index=False)

    return gaia_data


def calculate_astdiff(fields, footprint, workdir, gaia_dr, cat_name_preffix='splus_cats/', cat_name_suffix='.fits'):
    """Calculate the astrometric differences between any SPLUS catalogue as long as the columns are properly named"""

    field_names = np.array([n.replace('_', '-') for n in footprint['NAME']])

    for tile in fields:
        if tile == 'fakename':
            print('this is a filler name')
        else:
            sra = footprint['RA'][field_names == tile]
            sdec = footprint['DEC'][field_names == tile]
            tile_coords = SkyCoord(ra=sra[0], dec=sdec[0], unit=(u.hour, u.deg), frame='icrs', equinox='J2000')

            gaia_cat_path = workdir + 'gaia_' + gaia_dr + '/' + tile + '_gaiacat.csv'
            if os.path.isfile(gaia_cat_path):
                gaia_data = ascii.read(gaia_cat_path, format='csv')
            else:
                gaia_data = get_gaia(workdir, tile_coords, tile, gaia_dr)

            # cathdu = 1
            # if cathdu == 2:
            # scat = fits.open(workdir + cat_name_preffix + tile + cat_name_suffix)[2].data
            scat = fits.open(workdir + cat_name_preffix + tile + cat_name_suffix)[1].data
            splus_coords = SkyCoord(ra=scat['RA'], dec=scat['DEC'], unit=(u.deg, u.deg))
            gaia_coords = SkyCoord(ra=gaia_data['RAJ2000'], dec=gaia_data['DEJ2000'], unit=(u.deg, u.deg))
            idx, d2d, d3d = splus_coords.match_to_catalog_3d(gaia_coords)
            separation = d2d < 5.0 * u.arcsec

            sample = (scat['r_auto'] > 13) & (scat['r_auto'] < 19)
            sample &= scat['PhotoFlagDet'] == 0
            sample &= scat['CLASS_STAR'] > 0.95 # MAR cat nao tem CLASS_STAR

            finalscat = scat[separation & sample]
            finalgaia = gaia_data[idx][separation & sample]

            abspm = abs(finalgaia['pmRA']) + abs(finalgaia['pmDE'])
            # get masked values in gaia
            mx = np.ma.masked_invalid(abspm)
            lmt = np.percentile(abspm[~mx.mask], 95)
            mask = (abspm < lmt) & ~mx.mask
            # calculate splus - gaia declination
            dediff = 3600. * (finalscat['DEC'][mask]*u.deg - np.array(finalgaia['DEJ2000'])[mask]*u.deg)
            # calculate splus - gaia ra
            radiff = (finalscat['RA'][mask] - finalgaia['RAJ2000'][mask]) * 3600.
            #radiff = np.cos(finalscat['DELTA_J2000']*u.deg)[mask] * finalscat['ALPHA_J2000'][mask] * 3600.
            #radiff -= np.cos(np.array(finalgaia['DEJ2000'])*u.deg)[mask] * np.array(finalgaia['RAJ2000'][mask]) * 3600.

            results_dir = workdir + 'results/'
            if not os.path.isdir(results_dir):
                os.mkdir(results_dir)

            d = {'radiff': radiff, 'dediff': dediff, 'abspm': abspm[mask]}
            results = pd.DataFrame(data=d)
            path_to_results = results_dir + tile + '_splus-gaiaDR3_diff.csv'
            results.to_csv(path_to_results, index=False)

    return

def plot_diffs(datatab):
    """plot results"""
    data = pd.read_csv(datatab)
    radiff = data['radiff']
    dediff = data['dediff']
    abspm = data['abspm']

    # stats
    mra = np.median(radiff)
    percra = np.percentile(radiff, [0.15, 2.5, 16, 50, 84, 97.5, 99.85])
    mde = np.median(dediff)
    percde = np.percentile(dediff, [0.15, 2.5, 16, 50, 84, 97.5, 99.85])

    # definitions for the axes
    left, width = 0.1, 0.6
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(figsize=(9, 8))

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)

    # the scatter plot:
    lbl = r'$N = %i$' % len(radiff)
    sc = ax_scatter.scatter(radiff, dediff, c=abspm, s=10, cmap='plasma', label=lbl)
    ax_scatter.grid()
    ax_scatter.legend(loc='upper right', handlelength=0, scatterpoints=1,
                      fontsize=12)
    # contour_pdf(radiff.value, dediff.value, ax=ax_scatter, nbins=20,
    #            percent=0.3, colors='r')
    # contour_pdf(radiff.value, dediff.value, ax=ax_scatter, nbins=20,
    #            percent=5, colors='orange')
    # contour_pdf(radiff.value, dediff.value, ax=ax_scatter, nbins=20,
    #            percent=32, colors='c')

    cb = plt.colorbar(sc, ax=ax_histy, pad=.02)
    cb.set_label(r'$|\mu|\ \mathrm{[mas\,yr^{-1}]}$', fontsize=20)
    # plt.setp(cb.get_xticklabels(), fontsize=14)
    cb.ax.tick_params(labelsize=14)

    # now determine nice limits by hand:
    binwidth = 0.05
    lim = np.ceil(np.abs([radiff, dediff]).max() / binwidth) * binwidth
    ax_scatter.set_xlim((-lim, lim))
    ax_scatter.set_ylim((-lim, lim))
    plt.setp(ax_scatter.get_xticklabels(), fontsize=14)
    plt.setp(ax_scatter.get_yticklabels(), fontsize=14)

    # plot stats
    ax_histx.axvline(percra[0], color='k', linestyle='dashed', lw=1, zorder=1)
    ax_histx.axvline(percra[1], color='k', linestyle='dashed', lw=1, zorder=1)
    ax_histx.axvline(percra[2], color='k', linestyle='dashed', lw=1, zorder=1)
    ax_histx.axvline(percra[3], color='k', linestyle='dashed', lw=1, zorder=1)
    ax_histx.axvline(percra[4], color='k', linestyle='dashed', lw=1, zorder=1)
    ax_histx.axvline(percra[5], color='k', linestyle='dashed', lw=1, zorder=1)
    ax_histx.axvline(percra[6], color='k', linestyle='dashed', lw=1, zorder=1)
    ax_histy.axhline(percde[0], color='k', linestyle='dashed', lw=1, zorder=1)
    ax_histy.axhline(percde[1], color='k', linestyle='dashed', lw=1, zorder=1)
    ax_histy.axhline(percde[2], color='k', linestyle='dashed', lw=1, zorder=1)
    ax_histy.axhline(percde[3], color='k', linestyle='dashed', lw=1, zorder=1)
    ax_histy.axhline(percde[4], color='k', linestyle='dashed', lw=1, zorder=1)
    ax_histy.axhline(percde[5], color='k', linestyle='dashed', lw=1, zorder=1)
    ax_histy.axhline(percde[6], color='k', linestyle='dashed', lw=1, zorder=1)

    # build hists
    bins = np.arange(-lim, lim + binwidth, binwidth)
    # xlbl = r'$\overline{\Delta\alpha} = %.3f$' % percra[3].value
    xlbl = r'$\widetilde{\Delta\alpha} = %.3f$' % percra[3]
    xlbl += '\n'
    xlbl += r'$\sigma = %.3f$' % np.std(radiff)
    xx, xy, _ = ax_histx.hist(radiff, bins=bins, label=xlbl,
                              alpha=0.8, zorder=10)
    ax_histx.legend(loc='upper right', handlelength=0, fontsize=12)
    # ylbl = r'$\overline{\Delta\delta} = %.3f$' % percde[3].value
    ylbl = r'$\widetilde{\Delta\delta} = %.3f$' % percde[3]
    ylbl += '\n'
    ylbl += r'$\sigma = %.3f$' % np.std(dediff)
    yx, yy, _ = ax_histy.hist(dediff, bins=bins, orientation='horizontal',
                              label=ylbl, alpha=0.8, zorder=10)
    ax_histy.legend(loc='upper right', handlelength=0, fontsize=12)

    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())

    # labels
    ax_scatter.set_xlabel(r'$\mathrm{\Delta\alpha\ [arcsec]}$', fontsize=20)
    ax_scatter.set_ylabel(r'$\mathrm{\Delta\delta\ [arcsec]}$', fontsize=20)

    figpath = datatab.split('.')[0] + '.png'
    print('saving fig', figpath)
    plt.savefig(figpath, format='png', dpi=360)
    plt.show()

    return


if __name__ == '__main__':
    # workdir = '/ssd/splus/MAR-gaia-astrometry/'
    # workdir = '/ssd/splus/iDR4_astrometry/'
    workdir = '/storage/splus/splusDR3-gaiaDR3-astrometry/'
    footprint = ascii.read(workdir + 'tiles_new_status.csv')
    fields = pd.read_csv(workdir + 'dr3_fields.csv')
    # field_name_suffix = '_R_dual.catalog'
    # field_name_suffix = '_R.detection.cat'
    # field_name_preffix = 'sex_'
    # field_name_suffix = '_R_dual.fits'
    gaia_dr = 'DR3'

    # calculate to all tiles at once
    num_procs = 2
    b = list(fields['NAME'][:1])
    num_fields = np.unique(b).size
    if num_fields % num_procs > 0:
        print('reprojecting', num_fields, 'fields')
        increase_to = int(num_fields / num_procs) + 1
        i = 0
        while i < (increase_to * num_procs - num_fields):
            b.append('fakename')
            i += 1
        else:
            print(num_fields, 'already fulfill the conditions')
    tiles = np.array(b).reshape((num_procs, int(np.array(b).size / num_procs)))
    print('calculating for a total of', tiles.size, 'fields')
    jobs = []
    print('creating', num_procs, 'jobs...')
    for tile in tiles:
        process = multiprocessing.Process(target=calculate_astdiff, args=(tile, footprint, workdir, gaia_dr))
        jobs.append(process)

    # calculate_astdiff(fields, footprint, workdir, gaia_dr,
    #                   cat_name_preffix=field_name_preffix, cat_name_suffix=field_name_suffix)

    # start jobs
    print('starting', num_procs, 'jobs!')
    for j in jobs:
        j.start()

    # check if any of the jobs initialized previously still alive
    # save resulting table after all are finished
    proc_alive = True
    while proc_alive:
        if any(proces.is_alive() for proces in jobs):
            proc_alive = True
            time.sleep(1)
        else:
            print('All jobs finished')
            proc_alive = False

    print('Done!')

    # to run only after finished all stacking
    # datatab = workdir + 'results/' + fields[0] + '_splus-gaiaDR3_diff.csv'
    #
    # plot_diffs(datatab)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
