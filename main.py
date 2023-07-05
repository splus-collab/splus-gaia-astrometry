# This module is meant to calculate the differences between the astrometry from S-PLUS to that
# of Gaia DR2 or DR3
# 2022-01-08: Expanding to compare any given photometric catalogue with Gaia
# Herpich F. R. 2022-12-20 fabiorafaelh@gmail.com
# GitHub: herpichfr
# ORCID: 0000-0001-7907-7884
# ---
# 2023-01-12: Adding multiprocessing to speed up the process
# 2023-01-13: Adding a function to calculate the astrometric differences between any SPLUS catalogue as long as the
# columns are properly named
# ---
# 2023-07-04: Changing parameters to run MAR columns
#

from statspack.statspack import contour_pdf
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
import glob
path_root = os.getcwd()
sys.path.append(str(path_root) + '/splus-gaia-astrometry')


class SplusGaiaAst(object):
    """
    Find astrometry differences between any observation and Gaia's.
    Gaia DR2 and DR3 are implemented
    """

    def __init__(self):
        self.workdir: str = './'
        # Gaia DR2 = 345; Gaia DR3 = 355
        self.gaia_dr = '355'
        self.cat_name_preffix: str = ''
        self.cat_name_suffix: str = ''
        self.cathdu: int = 1
        self.racolumn: str = 'RA'
        self.decolumn: str = 'DEC'
        self.mag_column: str = 'MAG_AUTO'
        self.flags_column = None
        self.clstar_column = None
        self.fwhm_column = None
        self.sn_column = None
        self.filetype = '.fits'
        self.angle: float = 1.0
        self.sn_limit: float = 10.

    def get_gaia(self, tile_coords, tilename, workdir=None, gaia_dr=None, angle=1.0):
        """
        Query Gaia photometry available at Vizier around a given centre.

        tile_coords : SkyCoord object
          Central coordinates of the catalogue

        tilename : string
          Name of the central tile to use to search for the individual catalogues

        workdir : string
          Workdir path. Default is None

        gaia_dr : str | float
          Gaia's catalogue number as registered at Vizier

        angle : float
          Radius to search around the central coordinates through Gaia's catalogue in Vizier

        :returns: DataFrame containing the data queried around the given coordinates
        :rtype: Pandas DataFrame containing the data queried around the given coordinates
        """

        workdir = self.workdir if workdir is None else workdir
        gaia_dr = self.gaia_dr if gaia_dr is None else gaia_dr
        angle = self.angle if angle is None else angle

        # query Vizier for Gaia's catalogue using gaia_dr number. gaia_dr number needs to be known beforehand
        print('querying gaia/vizier')
        v = Vizier(columns=['*', 'RAJ2000', 'DEJ2000'],
                   catalog='I/' + str(gaia_dr))
        v.ROW_LIMIT = 999999999
        # change cache location to workdir path to avoid $HOME overfill
        cache_path = os.path.join(workdir, '.astropy/cache/astroquery/Vizier/')
        if not os.path.isdir(cache_path):
            try:
                os.makedirs(cache_path, exist_ok=True)
            except FileExistsError:
                print('File', cache_path, 'already exists. Skipping')
        v.cache_location = cache_path
        gaia_data = v.query_region(tile_coords, radius=Angle(angle, "deg"))[0]
        # mask all nan objects in the coordinates columns before saving the catalogue
        mask = gaia_data['RAJ2000'].mask & gaia_data['DEJ2000'].mask
        gaia_data = gaia_data[~mask]
        print('gaia_data is', gaia_data)

        # save Gaia's catalogue to workdir
        gaia_cat_path = os.path.join(
            workdir, 'gaia_' + gaia_dr + '/' + tilename + '_gaiacat.csv')
        if not os.path.isdir(workdir + 'gaia_' + gaia_dr):
            try:
                os.mkdir(workdir + 'gaia_' + gaia_dr)
            except FileExistsError:
                print('File', workdir + 'gaia_' +
                      gaia_dr, 'already exists. Skipping')

        print('saving result of match with gaia to', gaia_cat_path)
        gaia_data.to_pandas().to_csv(gaia_cat_path, index=False)

        return gaia_data

    def calculate_astdiff(self, fields, footprint, workdir=None, gaia_dr=None, cat_name_preffix=None,
                          cat_name_suffix=None):
        """
        Calculate the astrometric differences between any SPLUS catalogue as long as the columns are properly named
        """

        gaia_dr = self.gaia_dr if gaia_dr is None else gaia_dr
        workdir = self.workdir if workdir is None else workdir
        cat_name_preffix = self.cat_name_preffix if cat_name_preffix is None else cat_name_preffix
        cat_name_suffix = self.cat_name_suffix if cat_name_suffix is None else cat_name_suffix

        # field_names = np.array([n.replace('_', '-')
        #                        for n in footprint['NAME']])
        field_names = np.array([n for n in footprint['NAME']])
        results_dir = workdir + 'results/'
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)

        for tile in fields:
            if tile == 'fakename':
                print('this is a filler name')
            else:
                path_to_results = results_dir + tile + '_mar-gaiaDR3_diff.csv'
                if os.path.isfile(path_to_results):
                    print('catalogue for tile', tile,
                          'already exists. Skipping')
                else:
                    sra = footprint['RA'][field_names == tile]
                    sdec = footprint['DEC'][field_names == tile]
                    tile_coords = SkyCoord(ra=sra[0], dec=sdec[0], unit=(
                        u.hour, u.deg), frame='icrs', equinox='J2000')

                    gaia_cat_path = workdir + 'gaia_' + gaia_dr + '/' + tile + '_gaiacat.csv'
                    if os.path.isfile(gaia_cat_path):
                        print('reading gaia cat from database')
                        gaia_data = ascii.read(gaia_cat_path, format='csv')
                    else:
                        gaia_data = self.get_gaia(tile_coords, tile)

                    if self.filetype == '.fits':
                        try:
                            scat = fits.open(
                                workdir + cat_name_preffix + tile + cat_name_suffix)[self.cathdu].data
                        except TypeError:
                            print(
                                'catalogue is not in fits format. Define the proper format of the default variable filetype')
                    elif self.filetype == '.csv':
                        try:
                            scat = pd.read_csv(
                                workdir + cat_name_preffix + tile + cat_name_suffix)
                        except TypeError:
                            print(
                                'catalogue is not in csv format. Define the proper format of the default variable filetype')
                    else:
                        raise TypeError(
                            'filetype for input catalogue not supported. Use .fits or .csv')

                    splus_coords = SkyCoord(
                        ra=scat[self.racolumn], dec=scat[self.decolumn], unit=(u.deg, u.deg))
                    gaia_coords = SkyCoord(
                        ra=gaia_data['RAJ2000'], dec=gaia_data['DEJ2000'], unit=(u.deg, u.deg))
                    idx, d2d, d3d = splus_coords.match_to_catalog_3d(
                        gaia_coords)
                    separation = d2d < 5.0 * u.arcsec

                    sample = (scat[self.mag_column] > 14.) & (
                        scat[self.mag_column] < 19.)
                    if self.flags_column is None:
                        print(
                            'FLAGS column not available. Skipping using flags to select objects')
                    else:
                        sample &= scat[self.flags_column] == 0
                    if self.clstar_column is None:
                        print(
                            'Not considering CLASS_STAR as an option to select objects')
                    else:
                        try:
                            # MAR cat nao tem CLASS_STAR
                            sample &= scat[self.clstar_column] > 0.95
                        finally:
                            Warning('Column for CLASS_STAR not found. Ignoring')
                    if self.fwhm_column is None:
                        print('Not considering FWHM as an option to select objects')
                    else:
                        sample &= scat[self.fwhm_column] * 3600 < 2.5
                    if self.sn_column is None:
                        print('Not considering SN as an option to select objects')
                    else:
                        sample &= scat[self.sn_column] > self.sn_limit

                    finalscat = scat[separation & sample]
                    finalgaia = gaia_data[idx][separation & sample]

                    abspm = abs(finalgaia['pmRA']) + abs(finalgaia['pmDE'])
                    # get masked values in gaia
                    mx = np.ma.masked_invalid(abspm)
                    lmt = np.percentile(abspm[~mx.mask], 95)
                    mask = (abspm < lmt) & ~mx.mask
                    # calculate splus - gaia declination
                    dediff = 3600. * \
                        (finalscat[self.decolumn][mask]*u.deg -
                         np.array(finalgaia['DEJ2000'])[mask]*u.deg)
                    # calculate splus - gaia ra
                    radiff = 3600 * (finalscat[self.racolumn][mask] - finalgaia['RAJ2000'][mask]) *\
                        np.cos(np.array(finalgaia['DEJ2000'])[mask] * u.deg)

                    d = {'RA': finalscat[self.racolumn][mask],
                         'DEC': finalscat[self.decolumn][mask],
                         'RAJ2000': finalgaia['RAJ2000'][mask],
                         'DEJ2000': finalgaia['DEJ2000'][mask],
                         'radiff': radiff,
                         'dediff': dediff,
                         'abspm': abspm[mask]}
                    results = pd.DataFrame(data=d)
                    print('saving results to', path_to_results)
                    results.to_csv(path_to_results, index=False)

        return


def plot_diffs(datatab, contour=False, colours=None, savefig=False):
    """plot results"""

    data = pd.read_csv(datatab)
    mask = (data['radiff'] > -10) & (data['radiff'] < 10)
    mask &= (data['dediff'] > -10) & (data['dediff'] < 10)

    radiff = data['radiff'][mask]
    dediff = data['dediff'][mask]
    abspm = data['abspm'][mask]

    # stats
    # mra = np.median(radiff)
    percra = np.percentile(radiff, [0.15, 2.5, 16, 50, 84, 97.5, 99.85])
    print('percentiles for RA:', percra)
    # mde = np.median(dediff)
    percde = np.percentile(dediff, [0.15, 2.5, 16, 50, 84, 97.5, 99.85])
    print('percentiles for DEC:', percde)

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
    print('starting plot...')
    sc = ax_scatter.scatter(radiff, dediff, c=abspm,
                            s=10, cmap='plasma', label=lbl)
    print('finished plot...')
    ax_scatter.grid()
    ax_scatter.legend(loc='upper right', handlelength=0, scatterpoints=1,
                      fontsize=12)
    if contour:
        print('calculating contours...')
        contour_pdf(radiff, dediff, ax=ax_scatter, nbins=100, percent=[0.3, 4.55, 31.7],
                    colors=colours)
        print('Done!')

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
    print('plotting histograms percentiles...')
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
    print('building histograms...')
    if radiff.size < 1000000:
        bins = np.arange(-lim, lim + binwidth, binwidth)
    else:
        bins = 1000
    # xlbl = r'$\overline{\Delta\alpha} = %.3f$' % percra[3].value
    xlbl = r'$\widetilde{\Delta\alpha} = %.3f$' % percra[3]
    xlbl += '\n'
    xlbl += r'$\sigma = %.3f$' % np.std(radiff)
    print('build RA histogram...')
    xx, xy, _ = ax_histx.hist(radiff, bins=bins, label=xlbl,
                              alpha=0.8, zorder=10)
    ax_histx.legend(loc='upper right', handlelength=0, fontsize=12)
    # ylbl = r'$\overline{\Delta\delta} = %.3f$' % percde[3].value
    ylbl = r'$\widetilde{\Delta\delta} = %.3f$' % percde[3]
    ylbl += '\n'
    ylbl += r'$\sigma = %.3f$' % np.std(dediff)
    print('build DEC histogram...')
    yx, yy, _ = ax_histy.hist(dediff, bins=bins, orientation='horizontal',
                              label=ylbl, alpha=0.8, zorder=10)
    ax_histy.legend(loc='upper right', handlelength=0, fontsize=12)

    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())

    # labels
    ax_scatter.set_xlabel(r'$\mathrm{\Delta\alpha\ [arcsec]}$', fontsize=20)
    ax_scatter.set_ylabel(r'$\mathrm{\Delta\delta\ [arcsec]}$', fontsize=20)

    if savefig:
        figpath = datatab.split('.')[0] + '.png'
        print('saving fig', figpath)
        plt.savefig(figpath, format='png', dpi=360)
    showfig = True
    if showfig:
        plt.show()
    else:
        plt.close()

    return


if __name__ == '__main__':
    get_gaia = True
    make_plot = True

    # calculate astrometric precision for MAR catalogues
    workdir = '/ssd/splus/astrocatalogs/'
    # workdir = '/storage/splus/splusDR4_auto-gaiaDR3-astrometry-cos/'

    if get_gaia:
        # initialize the class
        gasp = SplusGaiaAst()

        # define default paths and additives
        gasp.workdir = workdir
        gasp.racolumn = 'ALPHA_J2000'
        gasp.decolumn = 'DELTA_J2000'
        gasp.cat_name_preffix = 'mar_cats/'
        gasp.cat_name_suffix = '.cat'
        gasp.mag_column = 'MAG_AUTO'
        gasp.flags_column = 'FLAGS'
        # gasp.clstar_column = 'CLASS_STAR'
        # gasp.sn_column = 's2n_r_psf'
        # gasp.sn_limit = 20.
        # gasp.fwhm_column = 'FWHM_R'
        gasp.cathdu = 2

        # read footprint table
        footprint = ascii.read(workdir + 'tiles_new_status.csv')
        # read list of fields to process
        fields = pd.read_csv(workdir + 'mar_fields.csv')

        # calculate to all tiles at once
        num_procs = 8
        b = list(fields['NAME'])
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
        tiles = np.array(b).reshape(
            (num_procs, int(np.array(b).size / num_procs)))
        print('calculating for a total of', tiles.size, 'fields')
        jobs = []
        print('creating', num_procs, 'jobs...')
        for tile in tiles:
            process = multiprocessing.Process(
                target=gasp.calculate_astdiff, args=(tile, footprint))
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

    if make_plot:
        # to run only after finished all stacking
        # datatab = workdir + 'results/results_stacked.csv'
        datatab = os.path.join(
            workdir, 'mar-astrometry_results_stacked.csv')
        if not os.path.isfile(datatab):
            list_results = glob.glob(
                workdir + 'results/*_mar-gaiaDR3_diff.csv')
            new_tab = pd.read_csv(list_results[0])
            for tab in list_results[1:]:
                print('stacking tab', tab, '...')
                t = pd.read_csv(tab)
                new_tab = pd.concat([new_tab, t], axis=0)
            print('saving results to', datatab)
            new_tab.to_csv(datatab, index=False)

        print('running plot module for table', datatab)
        plot_diffs(datatab, contour=False, colours=[
                   'limegreen', 'yellowgreen', 'c'], savefig=True)
