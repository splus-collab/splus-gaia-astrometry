# package with modules for statistic visualization
# herpich 2022-12-20 fabiorafaelh@gmail.com

import numpy as np
import scipy
import scipy.stats

def bining(x, y, z, nbins = 10, xlim = (None, None), ylim = (None, None),
           zlim = (None, None)):
    xv = np.linspace(xlim[0], xlim[1], nbins + 1)
    yv = np.linspace(ylim[0], ylim[1], nbins + 1)
    X, Y, Z = [], [], []
    for i in range(nbins):
        maskx = (x >= xv[i]) & (x < xv[i + 1])
        if len(x[maskx]) > 1:
            x_med = np.median(x[maskx])
        else:
            x_med = np.nan
        for k in range(nbins):
            masky = (y >= yv[k]) & (y < yv[k + 1])
            if len(y[maskx & masky]) > 1:
                y_med = np.median(y[maskx & masky])
                z_med = np.median(z[maskx & masky])
            else:
                y_med = np.nan
                z_med = np.nan
            X.append(x_med)
            Y.append(y_med)
            Z.append(z_med)
    return X, Y, Z

def find_confidence_interval(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level

def density_contour(xdata, ydata, binsx, binsy, ax = None, range = None,
                    fill = False, levels_prc = [.68, .95, .99],
                    **contour_kwargs):
    """ Create a density contour plot.

   Parameters
   ----------
   xdata : numpy.ndarray
   ydata : numpy.ndarray
   binsx : int
       Number of bins along x dimension
   binsy : int
       Number of bins along y dimension
   ax : matplotlib.Axes (optional)
       If supplied, plot the contour to this axis. Otherwise, open a new figure
   contour_kwargs : dict
       kwargs to be passed to pyplot.contour()
   """
    #nbins_x = len(binsx) - 1
    #nbins_y = len(binsy) - 1
    import scipy.optimize as so

    H, xedges, yedges = np.histogram2d(xdata, ydata, bins = [binsx, binsy], normed = True)
    x_bin_sizes = (xedges[1:] - xedges[:-1])
    y_bin_sizes = (yedges[1:] - yedges[:-1])

    pdf = (H * (x_bin_sizes * y_bin_sizes))

    levels = [ so.brentq(find_confidence_interval, 0., 1., args = (pdf, prc)) for prc in levels_prc ]
#    one_sigma = so.brentq(find_confidence_interval, 0., 1., args = (pdf, 0.90))
#    one_sigma = so.brentq(find_confidence_interval, 0., 1., args = (pdf, 0.68))
#    two_sigma = so.brentq(find_confidence_interval, 0., 1., args = (pdf, 0.95))
#    three_sigma = so.brentq(find_confidence_interval, 0., 1., args = (pdf, 0.99))
#    levels = [one_sigma]#, two_sigma, three_sigma]

    X, Y = 0.5 * (xedges[1:] + xedges[:-1]), 0.5 * (yedges[1:] + yedges[:-1])
    Z = pdf.T

    if ax == None:
        contour = plt.contour(X, Y, Z, levels = levels, origin = "lower",
                              **contour_kwargs)
        out = contour
        if fill == True:
            contourf = plt.contourf(X, Y, Z, levels = levels, origin = "lower",
                                   **contour_kwargs)
            out = contour, contourf
    else:
        contour = ax.contour(X, Y, Z, levels = levels, origin = "lower",
                             **contour_kwargs)
        out = contour
        if fill == True:
            contourf = ax.contourf(X, Y, Z, levels = levels, origin = "lower",
                                  **contour_kwargs)
            out = contour, contourf

    return out

def contour_pdf(x_axis, y_axis, ax=None, nbins=10, percent=10, colors='b'):
    '''
        contornos para percentis tirei deste site:
        http://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
    '''
    x1 = x_axis, y_axis
    xmin = min(x_axis)
    xmax = max(x_axis)
    ymin = min(y_axis)
    ymax = max(y_axis)
    xf = np.transpose(x1)
    pdf = scipy.stats.kde.gaussian_kde(xf.T)
    q, w = np.meshgrid(np.linspace(xmin, xmax, nbins),
                       np.linspace(ymin, ymax, nbins))
    r = pdf([q.flatten(), w.flatten()])
    s = scipy.stats.scoreatpercentile(pdf(pdf.resample(1000)), percent)
    r.shape = (nbins, nbins)
    if ax == None:
        contour = plt.contour(np.linspace(xmin, xmax, nbins),
                              np.linspace(ymin, ymax, nbins),
                              r, [s], linewidths=1.5, colors=colors)
    else:
        contour = ax.contour(np.linspace(xmin, xmax, nbins),
                             np.linspace(ymin, ymax, nbins),
                             r, [s], linewidths=1.5, colors=colors)

    return contour