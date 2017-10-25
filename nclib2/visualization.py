#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NCLib2 errors are here
@author: Milos.Korenciak@solargis.com
"""
from __future__ import print_function  # Python 2 vs. 3 compatibility --> use print()
# from __future__ import division  # Python 2 vs. 3 compatibility --> / returns float
from __future__ import unicode_literals  # Python 2 vs. 3 compatibility --> / returns float
from __future__ import absolute_import  # Python 2 vs. 3 compatibility --> absolute imports

import numpy as np

# logging
from .utils import make_logger
logger = make_logger(__name__)

def visualize_map_3d(map_data_3d, bbox, vmin=None, vmax=None, interpolation='bilinear', img_width=10, img_height=8,
                     show_grid=True, title=None, color='jet', countries_color='#999999', coast_color='#bbbbbb',
                     openfile=None, grid_step=20, subplot_titles_list=[], ocean_mask=False, resolution='i', num=1):
    from pylab import colorbar, show
    from matplotlib.widgets import Button
    from matplotlib import cm
    from pylab import axes, draw, figure
    import matplotlib.pyplot as plt

    try:
        from mpl_toolkits.basemap import Basemap
    except:
        from matplotlib.toolkits.basemap import Basemap

    predefined_color_ramp_dict = plt.cm.datad

    if color in predefined_color_ramp_dict.keys():
        cmap = plt.get_cmap(color)
    else:
        cmap = cm.jet

    fig = figure(num=num, figsize=(img_width, img_height), facecolor='w')

    map_data_3d_ma = np.ma.masked_where(np.isnan(map_data_3d), map_data_3d)

    if vmin is None:
        vmin = map_data_3d_ma.min()
    if vmax is None:
        vmax = map_data_3d_ma.max()
    vmin = vmin - 0.1 * (vmax - vmin)
    vmax = vmax + 0.1 * (vmax - vmin)

    m = Basemap(projection='cyl', llcrnrlon=bbox.xmin, llcrnrlat=bbox.ymin, urcrnrlon=bbox.xmax, urcrnrlat=bbox.ymax,
                resolution=resolution)
    if show_grid:
        m.drawparallels(np.arange(-90., 90., grid_step), labels=[1, 0, 0, 0], color='k')
        m.drawmeridians(np.arange(-180., 360., grid_step), labels=[0, 0, 0, 1], color='k')
    map_data = np.flipud(map_data_3d_ma[0, :, :])
    img = m.imshow(map_data, vmin=vmin, vmax=vmax, extent=(bbox.xmin, bbox.xmax, bbox.ymax, bbox.ymin),
                   interpolation=interpolation, cmap=cmap)
    m.drawcoastlines(color=coast_color)
    m.drawcountries(color=countries_color)
    if ocean_mask:
        m.drawlsmask(land_color='None', ocean_color='w')
    colorbar()

    # title
    if title is not None:
        fig.suptitle(title, fontsize=14)

    axprev = axes([0.85, 0.05, 0.06, 0.060])
    axnext = axes([0.92, 0.05, 0.06, 0.060])
    bnext = Button(axnext, '>')
    bprev = Button(axprev, '<')

    if len(subplot_titles_list) == map_data_3d.shape[0]:
        txt = fig.text(0.91, 0.02, '%d/%d: %s' % (1, map_data_3d.shape[0], subplot_titles_list[0]), size=8,
                       ha="center")
    else:
        txt = fig.text(0.91, 0.02, '%d/%d' % (1, map_data_3d.shape[0]), size=8, ha="center")

    class Index:
        ind = 0

        def next(self, event):
            self.ind += 1
            i = self.ind % map_data_3d_ma.shape[0]
            img_data = np.flipud(map_data_3d_ma[i, :, :])
            img.set_data(img_data)
            if len(subplot_titles_list) == map_data_3d.shape[0]:
                txt.set_text('%d/%d: %s' % (i + 1, map_data_3d.shape[0], subplot_titles_list[i]))
            else:
                txt.set_text('%d/%d' % (i + 1, map_data_3d.shape[0]))
            draw()

        def prev(self, event):
            self.ind -= 1
            i = self.ind % map_data_3d_ma.shape[0]
            img_data = np.flipud(map_data_3d_ma[i, :, :])
            img.set_data(img_data)
            if len(subplot_titles_list) == map_data_3d.shape[0]:
                txt.set_text('%d/%d: %s' % (i + 1, map_data_3d.shape[0], subplot_titles_list[i]))
            else:
                txt.set_text('%d/%d' % (i + 1, map_data_3d.shape[0]))
            draw()

    callback = Index()
    bnext.on_clicked(callback.next)
    bprev.on_clicked(callback.prev)

    if openfile is not None:
        fig.canvas.print_png(openfile)
    else:
        show()


# def show_raw_3d(d3_data):  # DEAD?
#     from pylab import colorbar, show
#     from matplotlib.widgets import Button
#     from matplotlib import cm
#     from pylab import axes, draw, figure
#     import matplotlib.pyplot as plt
#
#     predefined_color_ramp_dict = plt.cm.datad
#
#     if color in predefined_color_ramp_dict.keys():
#         cmap = plt.get_cmap(color)
#     else:
#         cmap = cm.jet
#
#     fig = figure(num=1, figsize=(img_width, img_height), facecolor='w')
#
#     map_data_3d_ma = np.ma.masked_where(np.isnan(map_data_3d), map_data_3d)
#
#     if vmin is None:
#         vmin = map_data_3d_ma.min()
#     if vmax is None:
#         vmax = map_data_3d_ma.max()
#     vmin = vmin - 0.1 * (vmax - vmin)
#     vmax = vmax + 0.1 * (vmax - vmin)
#
#     img = m.imshow(map_data, vmin=vmin, vmax=vmax, extent=(bbox.xmin, bbox.xmax, bbox.ymax, bbox.ymin),
#                    interpolation=interpolation, cmap=cmap)
#     m.drawcoastlines(color=coast_color)
#     m.drawcountries(color=countries_color)
#     if ocean_mask:
#         m.drawlsmask(land_color='None', ocean_color='w')
#     colorbar()
#
#     # title
#     if title is not None:
#         fig.suptitle(title, fontsize=14)
#
#     axprev = axes([0.85, 0.05, 0.06, 0.060])
#     axnext = axes([0.92, 0.05, 0.06, 0.060])
#     bnext = Button(axnext, '>')
#     bprev = Button(axprev, '<')
#
#     if len(subplot_titles_list) == map_data_3d.shape[0]:
#         txt = fig.text(0.91, 0.02, '%d/%d: %s' % (1, map_data_3d.shape[0], subplot_titles_list[0]), size=8,
#                        ha="center")
#     else:
#         txt = fig.text(0.91, 0.02, '%d/%d' % (1, map_data_3d.shape[0]), size=8, ha="center")
#
#     class Index:
#         ind = 0
#
#         def next(self, event):
#             self.ind += 1
#             i = self.ind % map_data_3d_ma.shape[0]
#             img_data = np.flipud(map_data_3d_ma[i, :, :])
#             img.set_data(img_data)
#             if len(subplot_titles_list) == map_data_3d.shape[0]:
#                 txt.set_text('%d/%d: %s' % (i + 1, map_data_3d.shape[0], subplot_titles_list[i]))
#             else:
#                 txt.set_text('%d/%d' % (i + 1, map_data_3d.shape[0]))
#             draw()
#
#         def prev(self, event):
#             self.ind -= 1
#             i = self.ind % map_data_3d_ma.shape[0]
#             img_data = np.flipud(map_data_3d_ma[i, :, :])
#             img.set_data(img_data)
#             if len(subplot_titles_list) == map_data_3d.shape[0]:
#                 txt.set_text('%d/%d: %s' % (i + 1, map_data_3d.shape[0], subplot_titles_list[i]))
#             else:
#                 txt.set_text('%d/%d' % (i + 1, map_data_3d.shape[0]))
#             draw()
#
#     callback = Index()
#     bnext.on_clicked(callback.next)
#     bprev.on_clicked(callback.prev)
#
#     if openfile is not None:
#         fig.canvas.print_png(openfile)
#     else:
#         show()



def show_raw(nd_array):
    """Simply show the data"""
    from matplotlib.pyplot import imshow, show
    imshow(nd_array)  # interpolation="nearest")
    show()
