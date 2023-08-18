#!/usr/bin/env python3
# This module is meant to calculate the differences between the astrometry from S-PLUS to that
# of Gaia DR2 or DR3
# 2022-01-08: Expanding to compare any given photometric catalogue with Gaia
# Herpich F. R. 2022-12-20 fabiorafaelh@gmail.com
# GitHub: herpichfr
# ORCID: 0000-0001-7907-7884
# ---

import os
import sys
import numpy as np
from astropy.io import ascii, fits
from astropy.table import vstack
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates import Angle
from astroquery.vizier import Vizier
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from datetime import datetime
import argparse
import logging
import colorlog
import git
from statspack import contour_pdf

__author__ = 'Fabio R Herpich'
__email__ = 'fabio.herpich@ast.cam.ac.uk'

__path__ = os.path.dirname(os.path.abspath(__file__))
repo = git.Repo(__path__)
try:
    latest_tag = repo.git.describe('--tags').lstrip('v')
    __version__ = latest_tag.lstrip('v')
except Exception:
    __version__ = 'unknown'


def parser():
    """
    Parse the arguments from the command line
    """

    parser = argparse.ArgumentParser(
        description='Calculate astrometric differences between S-PLUS and Gaia DR2 or DR3')
    parser.add_argument('-t', '--tiles', type=str,
                        required=True, help='List of tiles to be processed')
    parser.add_argument('-f', '--footprint', type=str, required=True,
                        help='Fooprint file containing the positions of the S-PLUS tiles.')
    parser.add_argument('-w', '--workdir', type=str, default=os.getcwd(),
                        help='Workdir path. Default is current directory',
                        required=False)
    parser.add_argument('-d', '--datadir', type=str, default=None,
                        help='Data directory path. Default is workdir',
                        required=False)
    # Gaia DR2 =345; Gaia DR3 = 355
    parser.add_argument('-g', '--gaia_dr', type=str, default='355',
                        help='Gaia catalogue number as registered at Vizier. Default is 355 (Gaia DR3)')
    parser.add_argument('-u', '--hdu', type=int, default=1,
                        help='HDU number of the catalogue when catalgue is FIST. Default is 1')
    parser.add_argument('-ra', '--racolumn', type=str, default='RA',
                        help='Column name of the RA in the catalogue. Default is RA')
    parser.add_argument('-de', '--deccolumn', type=str, default='DEC',
                        help='Column name of the DEC in the catalogue. Default is DEC')
    parser.add_argument('-m', '--mag_column', type=str, default='MAG_AUTO',
                        help='Column name of the magnitude in the catalogue. Default is MAG_AUTO')
    parser.add_argument('-fl', '--flags_column', type=str, default=None,
                        help='Column name of the flags in the catalogue. Default is None')
    parser.add_argument('-cs', '--clstar_column', type=str, default=None,
                        help='Column name of the clstar in the catalogue. Default is None')
    parser.add_argument('-fwhm', '--fwhm_column', type=str, default=None,
                        help='Column name of the fwhm in the catalogue. Default is None')
    parser.add_argument('-sn', '--sn_column', type=str, default=None,
                        help='Column name of the sn in the catalogue. Default is None')
    parser.add_argument('-a', '--angle', type=float, default=1.0,
                        help='Radius to search Gaia around the tile centre. Default is 1.0 deg')
    parser.add_argument('-sl', '--sn_limit', type=float, default=10.0,
                        help='Signal-to-noise lower limit to be used in the crossmatch. Default is 10.0')
    parser.add_argument('-o', '--output', type=str, default='results_stacked',
                        help='Output name of the stacked catalogue. Default is results_stacked.csv')
    parser.add_argument('-b', '--bins', type=int, default=1000,
                        help='Number of bins in the histogram. Default is 1000')
    parser.add_argument('-l', '--limits', type=float, default=0.05,
                        help='Limit of the histogram. Default is 0.5')
    parser.add_argument('-nc', '--ncores', type=int, default=1,
                        help='Number of cores to be used. Default is 1')
    parser.add_argument('--contour', action='store_true',
                        help='Plot the contour of the PDF. Default is False')
    parser.add_argument('--colours', type=list, default=['limegreen', 'yellowgreen', 'c'],
                        help="Colours of the histograms. Default is ['limegreen', 'yellowgreen', 'c']")
    parser.add_argument('--percents', type=str, default='[0.3,4.5,32]',
                        help="Percentiles of the contours (include the values without space separator). Default is 3, 2 and 1 sigma, or '[0.3,4.5,32]'")
    parser.add_argument('-sf', '--savefig', action='store_true',
                        help='Save the figure. Default is False')
    parser.add_argument('--showfig', action='store_true', default=True,
                        help='Save the figure. Default is False')
    parser.add_argument('--debug', action='store_true',
                        help='Prints out the debug of the code. Default is False')
    parser.add_argument('-vv', '--verbose', action='store_true',
                        help='Prints out the progress of the code. Default is False')
    parser.add_argument('--clobber', action='store_true',
                        help='Overwrite the output file. Default is False')

    if len(sys.argv) == 1:
        parser.print_help()
        raise argparse.ArgumentTypeError(
            'No arguments provided. Showing the help message.')

    args = parser.parse_args()

    return args


class SplusGaiaAst(object):

    def __init__(self, args):
        # atrgs from parser
        self.tiles: str = args.tiles
        self.footprint: str = args.footprint
        self.workdir: str = args.workdir
        self.datadir: str = args.datadir
        self.gaia_dr = args.gaia_dr
        self.cathdu: int = args.hdu
        self.racolumn: str = args.racolumn
        self.decolumn: str = args.deccolumn
        self.mag_column: str = args.mag_column
        self.flags_column = args.flags_column
        self.clstar_column = args.clstar_column
        self.fwhm_column = args.fwhm_column
        self.sn_column = args.sn_column
        self.angle: float = args.angle
        self.sn_limit: float = args.sn_limit
        self.output: str = args.output
        self.bins = args.bins
        self.limit = args.limits
        self.ncores = args.ncores
        self.savefig: bool = args.savefig
        self.debug: bool = args.debug
        self.verbose: bool = args.verbose
        self.clobber: bool = args.clobber

        # other attributes
        self.logger = logging.getLogger(__name__)
        self.data_dict = {}
        self.foot_table = None

    def execute(self):
        """
        Execute the code
        """
        # Load the tiles to be used
        fields = self.get_fields()

        # loag the footprint
        self.foot_table = self.get_footprint()

        # Load the data
        self.data_dict = self.load_data(fields)

        # calc gaia and splus astrometry difference using ncores to parallelize
        pool = mp.Pool(processes=self.ncores)
        splus_gaia_astdiff = pool.map(
            self.calculate_astdiff, list(self.data_dict.keys()))
        pool.close()
        pool.join()
        return splus_gaia_astdiff

    def get_fields(self):
        """
        Get the fields from the tiles

        Parameters
        ----------
        tiles : str
          Tiles to be used

        Returns
        -------
        fields : list
          List of fields
        """
        # read text file with the list of tiles to consider
        textfile_path = os.path.join(self.workdir, self.tiles)
        assert os.path.exists(textfile_path), 'File {} does not exist'.format(
            textfile_path)
        fields = pd.read_csv(textfile_path, sep=' ',
                             header=None, names=['NAME'])

        return fields

    def get_footprint(self):
        """
        Get the footprint of the survey

        Parameters
        ----------
        footprint : str
            Path to the footprint file

        Returns
        -------
        footprint : astropy Table
            Table containing the footprint of the survey
        """
        path_to_foot = os.path.abspath(self.footprint)
        try:
            footprint = ascii.read(path_to_foot)
        except FileNotFoundError:
            self.logger.error(" - ".join([datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S'), 'Footprint file {} not found'.format(path_to_foot)]))
            raise FileNotFoundError(
                "Footprint file {} not found".format(path_to_foot))

        return footprint

    def get_fields_names(self, footprint):
        """
        Get the names of the fields in the footprint and correct them is necessary
        """

        try:
            field_names = np.array([n.replace('_', '-')
                                    for n in footprint['NAME']])
        except ValueError:
            field_names = footprint['NAME']

        return field_names

    def load_data(self, fields):
        """
        Load the data from the tiles

        Parameters
        ----------
        fields : list
          List of fields

        Returns
        -------
        data : list
          List to tables inside indir
        """
        data_dict = {}
        for item in os.listdir(self.datadir):
            if item.split('.')[0] in fields['NAME'].values:
                data_dict[item.split('.')[0]] = os.path.join(
                    self.datadir, item)

        return data_dict

    def calculate_astdiff(self, tile):
        """
        Calculate the astrometric differences between any SPLUS catalogue as
        long as the columns are properly named

        Parameters
        ----------
        tile : str
            Name of the tile

        Returns
        -------
        astrometry : astropy Table
            Table containing the astrometric differences between the SPLUS catalogues and Gaia
        """

        gaia_dr = self.gaia_dr
        workdir = self.workdir

        # create the results directory if it doesn't exist
        results_dir = os.path.join(workdir, 'results/')
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)

        path_to_results = os.path.join(
            results_dir, "".join([tile, '_gaiaDR_diff.csv']))
        if os.path.isfile(path_to_results):
            self.logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                         'Catalogue for tile %s already exists. Skipping calculation' % tile]))
            results = ascii.read(path_to_results, format='csv')

            return results
        else:
            sra = self.foot_table['RA'][self.foot_table['NAME'] == tile]
            sdec = self.foot_table['DEC'][self.foot_table['NAME'] == tile]
            tile_coords = SkyCoord(ra=sra[0], dec=sdec[0], unit=(
                u.hour, u.deg), frame='icrs', equinox='J2000')

            gaia_cat_path = os.path.join(workdir, "".join(
                ['gaia_', gaia_dr, '/', tile, '.csv']))
            if os.path.isfile(gaia_cat_path):
                self.logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                             'Reading gaia cat from database']))
                gaia_data = ascii.read(gaia_cat_path, format='csv')
            else:
                gaia_data = self.get_gaia(tile_coords, tile)

            # test if input catalogue is in FITS or CSV format
            try:
                scat = fits.open(os.path.join(workdir, self.data_dict[tile]))[
                    self.cathdu].data
                self.logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                             'Catalogue is in FITS format. Reading hdu %i' % self.cathdu]))
            except OSError:
                scat = ascii.read(os.path.join(
                    workdir, self.data_dict[tile]))
                self.logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                             'Catalogue is in CSV format']))
            except (UnicodeDecodeError, TypeError):
                self.logger.error(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                              'Filetype for input catalogue not supported. Use FITS or CSV']))
                raise TypeError(
                    'Filetype for input catalogue not supported')

            splus_coords = SkyCoord(
                ra=scat[self.racolumn], dec=scat[self.decolumn], unit=(u.deg, u.deg))
            gaia_coords = SkyCoord(
                ra=gaia_data['RAJ2000'], dec=gaia_data['DEJ2000'], unit=(u.deg, u.deg))
            idx, d2d, _ = splus_coords.match_to_catalog_3d(
                gaia_coords)
            separation = d2d < 5.0 * u.arcsec

            sample = (scat[self.mag_column] > 14.)
            sample &= (scat[self.mag_column] < 19.)
            if self.flags_column is None:
                self.logger.warning(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                'FLAGS column not available. Skipping using flags to object selection']))
            else:
                sample &= scat[self.flags_column] == 0
            if self.clstar_column is None:
                self.logger.warning(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                'CLASS_STAR column not available. Skipping using CLASS_STAR to object selection']))
            else:
                try:
                    sample &= scat[self.clstar_column] > 0.95
                except:
                    self.logger.warning(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                    'Column for CLASS_STAR not found. Ignoring']))
            if self.fwhm_column is None:
                self.logger.warning(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                'FWHM column not available. Skipping using FWHM to object selection']))
            else:
                sample &= scat[self.fwhm_column] * 3600 < 2.5
            if self.sn_column is None:
                self.logger.warning(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                'SN column not available. Skipping using SN to object selection']))
            else:
                sample &= scat[self.sn_column] > self.sn_limit

            finalscat = scat[separation & sample]
            finalgaia = gaia_data[idx][separation & sample]

            abspm = abs(finalgaia['pmRA']) + abs(finalgaia['pmDE'])
            # get masked values in gaia
            mx = np.ma.masked_invalid(abspm)
            lmt = np.percentile(abspm[~mx.mask], 95)
            mask = (abspm < lmt) & ~mx.mask
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
            self.logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                         'Saving results to %s' % path_to_results]))
            results.to_csv(path_to_results, index=False)

            return results

    def get_gaia(self, tile_coords, tilename, workdir=None, gaia_dr=None, angle=1.0):
        """
        Query Gaia photometry available at Vizier around a given centre.

        Parameters
        ----------
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

        Returns
        -------
        gaia : Pandas DataFrame
            DataFrame containing the data queried around the given coordinates
        """
        workdir = workdir if self.workdir is None else self.workdir
        gaia_dr = gaia_dr if self.gaia_dr is None else self.gaia_dr
        angle = angle if self.angle is None else self.angle

        # query Vizier for Gaia's catalogue using gaia_dr number. gaia_dr number needs to be known beforehand
        self.logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                     'Querying gaia/vizier for tile %s' % tilename]))
        v = Vizier(columns=['*', 'RAJ2000', 'DEJ2000'],
                   catalog='I/' + str(gaia_dr))
        v.ROW_LIMIT = 999999999
        # change cache location to workdir path to avoid $HOME overfill
        cache_path = os.path.join(workdir, '.astropy/cache/astroquery/Vizier/')
        if not os.path.isdir(cache_path):
            try:
                os.makedirs(cache_path, exist_ok=True)
            except FileExistsError:
                self.logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                             "File %s already exists. Skipping", cache_path]))
        v.cache_location = cache_path
        gaia_data = v.query_region(tile_coords, radius=Angle(angle, "deg"))[0]
        # mask all nan objects in the coordinates columns before saving the catalogue
        mask = gaia_data['RAJ2000'].mask & gaia_data['DEJ2000'].mask
        gaia_data = gaia_data[~mask]
        if self.verbose:
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                  'Gaia_data is %s' % gaia_data)

        # save Gaia's catalogue to workdir
        gaia_cat_path = os.path.join(workdir, "".join(
            ['gaia_', gaia_dr, '/', tilename, '_gaiacat.csv']))
        if not os.path.isdir(os.path.join(workdir, " - ".join(['gaia_', gaia_dr]))):
            try:
                os.mkdir(os.path.join(workdir, "".join(['gaia_', gaia_dr])))
            except FileExistsError:
                self.logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                             "File %s already exists. Skipping",
                                             os.path.join(workdir, "".join(['gaia_', gaia_dr]))]))
        if self.verbose:
            self.logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                         'Saving gaia catalogue to cache %s', gaia_cat_path]))
        gaia_data.to_pandas().to_csv(gaia_cat_path, index=False)

        return gaia_data


def plot_diffs(datatab, args):
    """
    Plots the differences between S-PLUS and Gaia.

    Parameters
    ----------
    datatab : str
        Path to the table with the differences.
    contour : bool, optional
        If True, plots the contours of the distribution.
    colours : str, optional
        Path to the file with the colours.
    savefig : bool, optional
        If True, saves the figure.
    """
    contour = args.contour
    percents = [float(x) for x in args.percents.strip('[]').split(',')]
    colours = args.colours
    savefig = args.savefig
    bins = args.bins
    limits = args.limits
    showfig = args.showfig
    verbose = args.verbose

    call_logger()
    logger = logging.getLogger('plot_diffs')
    
    if not os.path.exists(datatab):
        datatab += ".csv"

    data = pd.read_csv(datatab)
    mask = (data['radiff'] > -10) & (data['radiff'] < 10)
    mask &= (data['dediff'] > -10) & (data['dediff'] < 10)

    radiff = data['radiff'][mask]
    dediff = data['dediff'][mask]
    abspm = data['abspm'][mask]

    percra = np.percentile(radiff, [0.15, 2.5, 16, 50, 84, 97.5, 99.85])
    logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'Percentiles for RA: %s' % percra]))
    percde = np.percentile(dediff, [0.15, 2.5, 16, 50, 84, 97.5, 99.85])
    logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'Percentiles for DEC: %s' % percde]))

    left, width = 0.1, 0.6
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    plt.figure(figsize=(9, 8))

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)

    lbl = r'$N = %i$' % len(radiff)
    logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Starting plot..."]))
    sc = ax_scatter.scatter(radiff, dediff, c=abspm,
                            s=10, cmap='plasma', label=lbl)
    logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Finished scatter plot..."]))
    ax_scatter.grid()
    ax_scatter.legend(loc='upper right', handlelength=0, scatterpoints=1,
                      fontsize=12)
    cb = plt.colorbar(sc, ax=ax_histy, pad=.02)
    cb.set_label(r'$|\mu|\ \mathrm{[mas\,yr^{-1}]}$', fontsize=20)
    cb.ax.tick_params(labelsize=14)

    # now determine nice limits by hand:
    binwidth = limits
    lim = np.ceil(np.abs([radiff, dediff]).max() / binwidth) * binwidth
    ax_scatter.set_xlim((-lim, lim))
    ax_scatter.set_ylim((-lim, lim))
    plt.setp(ax_scatter.get_xticklabels(), fontsize=14)
    plt.setp(ax_scatter.get_yticklabels(), fontsize=14)

    if verbose:
        logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Plotting histograms for the percentiles...",]))
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
    if verbose:
        logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Building histograms..."]))
    if radiff.size < 1000000:
        bins = np.arange(-lim, lim + binwidth, binwidth)
    else:
        bins = 1000
    xlbl = "".join([r'$\overline{\Delta\alpha} = %.3f$' % percra[3], '\n',
                    r'$\sigma = %.3f$' % np.std(radiff)])
    if verbose:
        logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Building RA histogram..."]))
    xx, xy, _ = ax_histx.hist(radiff, bins=bins, label=xlbl,
                              alpha=0.8, zorder=10)
    ax_histx.legend(loc='upper right', handlelength=0, fontsize=12)
    ylbl = "".join([r'$\overline{\Delta\delta} = %.3f$' % percde[3], '\n',
                    r'$\sigma = %.3f$' % np.std(dediff)])
    if verbose:
        logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Building DEC histogram...",]))
    yx, yy, _ = ax_histy.hist(dediff, bins=bins, orientation='horizontal',
                              label=ylbl, alpha=0.8, zorder=10)
    ax_histy.legend(loc='upper right', handlelength=0, fontsize=12)

    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())

    # labels
    ax_scatter.set_xlabel(r'$\mathrm{\Delta\alpha\ [arcsec]}$', fontsize=20)
    ax_scatter.set_ylabel(r'$\mathrm{\Delta\delta\ [arcsec]}$', fontsize=20)

    if contour:
        logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Calculatinig contours..."]))
        contour_pdf(radiff, dediff, ax=ax_scatter, nbins=200,
                    percent=percents, colors=colours)
        if verbose:
            logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "Done!"]))
        logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Done!"]))

    # restart logger
    call_logger()
    logger = logging.getLogger('plot_diffs')

    if savefig:
        figpath = datatab.split('.')[0] + '.png'
        logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Saving figure at %s" % figpath]))
        plt.savefig(figpath, format='png', dpi=360)

    if showfig:
        if verbose:
            logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "Showing figure..."]))
        plt.show()
    else:
        if verbose:
            logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "Closing figure..."]))
        plt.close()

    return


def call_logger():
    """Configure the logger."""
    logging.shutdown()
    logging.root.handlers.clear()

    # configure the module with colorlog
    logger = colorlog.getLogger()
    logger.setLevel(logging.INFO)

    # create a formatter with green color for INFO level
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)s:%(name)s:%(message)s',
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'blue',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        })

    # create handler and set the formatter
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    # add the handler to the logger
    logger.addHandler(ch)


if __name__ == '__main__':
    call_logger()
    # get the path where the code resides
    code_path = os.path.dirname(os.path.abspath(__file__))
    # get the arguments passed from the command line
    args = parser()
    gasp = SplusGaiaAst(args)
    gasp.datadir = args.datadir if args.datadir is not None else args.workdir

    list_of_matches = gasp.execute()
    if list_of_matches is None:
        gasp.logger.error(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                      "No matches found. Exiting...",]))
        sys.exit(1)
    else:
        file_to_save = os.path.join(
            args.workdir, ''.join([args.output, '.csv']))
        if os.path.isfile(file_to_save) and not args.clobber:
            gasp.logger.warning(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            "File %s already exists. Use --clobber to force overwrite." %
                                            file_to_save]))
        else:
            gasp.logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                         "Found %d matches. Starting staking" %
                             len(list_of_matches)]))
            # stack the results
            stacked_results = vstack(list_of_matches)
            gasp.logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                         "Saving results to %s" % file_to_save]))
            stacked_results.write(file_to_save, format='csv', overwrite=True)

    datatab = os.path.join(args.workdir, args.output)

    gasp.logger.info(" - ".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     'Running plot module for table', datatab]))
    plot_diffs(datatab, args)
