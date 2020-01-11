# survey.py
# ed
# methods for dealing with an acoustic telemetry survey
from __future__ import print_function

import sys, os
import numpy as np
from collections import defaultdict
import pylab
from osgeo import ogr
import pandas as pd
import track

import six
six.moves.reload_module(track)
from track import Track

from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import stompy.grid.unstructured_grid as ugrid
import stompy.plot.cmap as cmap
from osgeo import ogr
import shapely.wkb
import shapely
from scipy.interpolate import interp1d

import pdb

cm_red = plt.get_cmap('autumn_r')

## 
# helper functions
def get_lines_from_shp(shp):
    ods = ogr.Open(shp)
    layer = ods.GetLayer(0)
    nlines = layer.GetFeatureCount()
    layer.ResetReading()
    lines = {}
    for i in range(nlines):
        feat = layer.GetNextFeature()
        if not feat:
            raise Exception("Expected more lines, but the layer stopped giving")
        
        fid = feat.GetFID()
        geo = feat.GetGeometryRef() 

        if geo.GetGeometryName() != 'LINESTRING':
            raise Exception("All features must be lines")

        # read the points into a numpy array:
        geom = shapely.wkb.loads(geo.ExportToWkb())
        points = np.array(geom.coords)
        name_field = 'name'
        index = feat.GetFieldIndex(name_field)
        name = feat.GetFieldAsString(index)
        lines[name] = points

    return lines

def truncate_colormap(cmap_in, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap_in.name, a=minval, b=maxval),
        cmap_in(np.linspace(minval, maxval, n)))
    return new_cmap

# somewhat hardwired flow csv read
def load_comma_formatted(data_fn, column_name):
    fp = open(data_fn,'rt')
    
    nheader = 1
    # read header
    for i in range(nheader):
        header = fp.readline().split(',')
    nc = header.index(column_name)
    print("nc: ",nc)
    dtime = []
    q = []
    
    for line in fp:
        try:
            fields = line.split(',')
            datestr = fields[1]
            dtime_tmp = datetime.strptime(datestr,'%d %b %Y %H:%M')
            dtime.append(dtime_tmp)
            q_cms = float(fields[nc])*0.0283168
            q.append(q_cms)
        except:
            break
        
    fp.close()
    return dtime, q

# segment intersection stuff from 
# https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)

# global variables
bathy_cmap = cmap.load_gradient('cyanotype_01.cpt')
cm_bath = truncate_colormap(bathy_cmap,0.1,0.9)
cmap_r = plt.get_cmap('autumn_r')
jet = plt.get_cmap('jet')
xylims = [[647100, 647500],[4185600, 4186000]]

class Survey(object):
    """ A survey consists of a set of acoustic telemetry tracks collected 
        around the same time.
    """

    def __init__(self, 
                 filename = None,
                 grd_file = None,
                 shp_file = None,
                 trackIDs = None):
        self.filename = filename
        self.load_grid(grd_file)
        self.load_polygons(shp_file)
        self.get_tracks(trackIDs=trackIDs)

    def get_tracks(self, trackIDs=None):
        """ Read all tracks in a survey """
        self.read_survey()
        if trackIDs == None:
            self.IDs = np.unique(self.dframe['ID'].values)
        else:
            self.IDs = trackIDs 
        self.ntracks = len(self.IDs)
        self.tracks = {}
        for ntag, ID in enumerate(self.IDs):
            dframe_track = self.dframe.loc[self.dframe['ID']==ID]
            # saves dataframe as track.df_track
            self.tracks[ID] = Track(dframe_track, grd=self.grd) 

    def set_directories(self, default_fig_dir='.', csv_dir='.'):
        self.default_fig_dir = default_fig_dir
        self.csv_dir = csv_dir

    def read_survey(self):
        """ read csv survey """
        self.dframe = pd.read_csv(self.filename, index_col=0)

    def read_receiver_locations(self, fname):
        self.rcv_dframe = pd.read_csv(fname, index_col=0)
        self.rcv_rec = self.rcv_dframe.to_records()

        return

    def load_grid(self, grd_file):
        self.grd = ugrid.PtmGrid(grd_file)
        self.grd.cells_area()

        return

    def load_polygons(self, shp_file):
        ods = ogr.Open(shp_file)
        layer = ods.GetLayer(0)
        npolygons = layer.GetFeatureCount()
        name_field = 'name'
        self.names = []
        self.polys = []
        for i in range(npolygons):
            feat = layer.GetNextFeature()
            name = str(feat.GetField(name_field))
            self.names.append(name)
            geo = feat.GetGeometryRef() 
            if geo.GetGeometryName() != 'POLYGON':
                raise Exception("Features must be polygons")
            poly = shapely.wkb.loads(geo.ExportToWkb())
            self.polys.append(poly)
 
        return

    def speed_over_ground(self, masked=False):
        """ estimate speed over ground for each track """
        if masked:
            self.goodsegs = {}
        else:
            self.segments = {}
        for nt, key in enumerate(self.tracks.keys()):
            tr = self.tracks[key]
            if masked:
                # check if "flagged" is set
                if 'flagged' not in tr.rec_track.dtype.names:
                    tr.total_outlier_flag()
                not_flagged = np.where(tr.rec_track.flagged==0)[0]
                rec_track = tr.rec_track[not_flagged]
            else:
                rec_track = tr.rec_track
            if len(rec_track) > 1:
                self.segments[key] = tr.make_segments(set_depth=True,
                                     input_rec_track=rec_track,
                                     overwrite_rec_seg=True)
               

        return
            
    def swimming_speed(self, hydro_fname=None, tracks=None):
        """ estimate speed over ground for each track """
        self.swim_dframe = pd.read_csv(hydro_fname, index_col=0)
        self.swim_IDs = np.unique(self.swim_dframe['id'].values)
        # assume that swim data exists for all tracks
        if tracks == None:
            tracks = self.tracks.keys()
        for nt, key in enumerate(tracks):
            tr = self.tracks[key]
            dframe_swim = self.swim_dframe.loc[self.swim_dframe['id']==key]
            tr.set_swim(dframe_swim)

        return
            
    def find_change_points(self, tracks=None):
        """ estimate change points of smoothed track """
        if tracks == None:
            tracks = self.tracks.keys()
        for nt, key in enumerate(tracks):
            tr = self.tracks[key]
            tr.find_change_points()

        return
            
    def find_change_points_fill(self, tracks=None):
        """ estimate change points of smoothed and filled track """
        if tracks == None:
            tracks = self.tracks.keys()
        for nt, key in enumerate(tracks):
            tr = self.tracks[key]
            tr.find_change_points_fill()

        return

    def calc_smoothed(self, masked=False, tracks=None, npts=True):
        """ calculate smoothed position and velocity """
        if tracks == None:
            tracks = self.tracks.keys()
        for nt, key in enumerate(tracks):
            tr = self.tracks[key]
            print("tr, nd",tr.ID, tr.ndetects)
            if tr.valid:
                if masked:
                    mask = tr.get_mask(methods=self.outlier_methods)
                else:
                    mask=None
                if npts:
                    tr.smoothed_vel_npts(mask=mask)
                    tr.smoothed_pos_npts(mask=mask)
                else:
                    tr.smoothed_vel(mask=mask)
                    tr.smoothed_pos(mask=mask)

        return

    def calc_autocorr(self, masked=False, tracks=None):
        if tracks == None:
            tracks = self.tracks.keys()
        for nt, key in enumerate(tracks):
            tr = self.tracks[key]
            if tr.valid:
                if masked:
                    mask = tr.get_mask(methods=self.outlier_methods)
                else:
                    mask=None
                tr.autocorr(mask=mask)

        return

    def plot_all_speed_vectors(self, variable='speed_over_ground'):
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.plot_bathy(ax)
        for nt, key in enumerate(self.tracks.keys()):
            tr = self.tracks[key]
            tr.plot_speed(variable, ax)
        self.format_map(ax)
        fig_name = os.path.join(self.default_fig_dir, 'all_vectors.png')
        plt.savefig(fig_name, dpi=800)

        return

    def plot_all_detects(self, tracks=None, daynight=False, routesel=False,
                         fig_name=None):
        if tracks == None:
            tracks = self.tracks.keys()
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.plot_bathy(ax)
        if daynight:
            color_by = 'daynight'
        elif routesel:
            color_by = 'routesel'
        else:
            color_by = 'age'
        for nt, key in enumerate(tracks):
            tr = self.tracks[key]
            not_flagged = np.where(tr.rec_track.flagged == 0)[0]
            tr.plot_detects(ax, mask=not_flagged, color_by=color_by)
        self.format_map(ax)
        if fig_name is None:
            fig_name = 'all_detects.png'
        fig_name = os.path.join(self.default_fig_dir, fig_name)
        plt.savefig(fig_name, dpi=800)

        return

    def plot_speed_single_track(self, variable='speed_over_ground', 
                                      color_by=None, time_series=False,
                                      histograms=False, plot_detects=True,
                                      show_metrics=False,
                                      fig_dir=None, masked=False,
                                      tracks=None,
                                      changepts=False):
        """ plot information related to a single track/tag """
        if time_series and histograms:
            print("Cannot select both time_series and histograms in speed plot")
            return
        if tracks == None:
            tracks = self.tracks.keys()

        for nt, key in enumerate(tracks):
            plt.close()
            fig = plt.figure()
            if time_series or histograms:
                gs = plt.GridSpec(4,4)
                ax = fig.add_subplot(gs[:-1,:])
            else:
                ax = fig.add_subplot(111)
            self.plot_bathy(ax)
            tr = self.tracks[key]
            not_flagged = np.where(tr.rec_track.flagged == 0)[0]
            tr.plot_speed(variable, ax, color_by=color_by, mask=not_flagged,
                          changepts=changepts)
            self.add_vel_scale(ax, tr)
            if plot_detects:
                tr.plot_detects(ax, color_by=color_by, plot_smoothed=False, 
                                plot_outliers=False, mask=not_flagged)
            self.format_map(ax)
            if show_metrics:
                self.show_metrics(ax, nt)
                nreds = len(self.reds)
                nyellows = len(self.yellows)
                if nreds > 0:
                    rstring = ''
                    for nr, red in enumerate(self.reds):
                        rstring = rstring + red
                        if nr < nreds - 1:
                            rstring = rstring + ', '
                    ax.set_title('%s: %s'%(key, rstring), color='r')
                elif nyellows > 0:
                    rstring = ''
                    for nr, yellow in enumerate(self.yellows):
                        rstring = rstring + yellow
                        if nr < nyellows - 1:
                            rstring = rstring + ', '
                    ax.set_title('%s: %s'%(key, rstring), color='yellow')
                else:
                    ax.set_title('%s: Good'%(key), color='g')

            if time_series:
                ax2 = fig.add_subplot(gs[-1,:])
                tr.plot_vel_ts(variable, ax2)
                ax2.yaxis.grid(True)

            if histograms:
                if variable == 'swimming_speed':
                    ax2 = fig.add_subplot(gs[-1,:2])
                    ax3 = fig.add_subplot(gs[-1,2:])
                    #tr.plot_swim_vel_histograms(ax2, ax3, masked=masked)
                    tr.plot_swim_vel_scatter(ax2, ax3, masked=masked, transpose_coords=True)
                    plt.subplots_adjust(wspace=3)
                else:
                    if masked:
                        turn_angle = tr.turn_angle(mask=not_flagged)
                    else:
                        turn_angle = tr.rec_track.turn_angle
                    ax2 = fig.add_subplot(gs[-1,:2])
                    ax3 = fig.add_subplot(gs[-1,2:])
                    tr.plot_del_vel_histograms(ax2, ax3, turn_angle, 
                                               masked=masked)
                    plt.subplots_adjust(wspace=3)
            # save figure
            if fig_dir == None:
                fig_dir = self.default_fig_dir
            fig_name = os.path.join(fig_dir, '%s_%s.png'%(key,variable))
            os.path.exists(fig_dir) or os.makedirs(fig_dir)
            plt.savefig(fig_name, dpi=800)

        return
        
    def identify_outliers(self, methods, params):
        self.outlier_methods = methods
        self.outlier_params = params
        for nt, key in enumerate(self.tracks.keys()):
            print(key)
            tr = self.tracks[key]
            tr.identify_outliers(methods, params)
#           if 'Iterative' in outlier_methods:
                #tr.total_outlier_flag()
                #not_flagged = np.where(tr.rec_track.flagged == 0)[0]
#               n = self.outlier_methods.index('Iterative')
#               max_del_vel = self.outlier_params[n][0]
#               tr.identify_outliers_iterative(max_del_vel)
            # add net flagged field to track
            tr.total_outlier_flag()
        
        return

    def write_to_csvs(self, fnames=None, masked=False, add_smooth=True):
        self.segments = {}
        for nt, key in enumerate(self.tracks.keys()):
            tr = self.tracks[key]
            self.segments[key] = tr.make_segments(set_depth=True)
            if masked:
                not_flagged = np.where(tr.rec_track.flagged == 0)[0]
                rec_tr = tr.rec_track[not_flagged]
                npts = len(rec_tr)
                if npts > 1:
                    rec_seg = tr.make_segments(input_rec_track=rec_tr)
                rec_swim = tr.rec_swim
            else: 
                rec_tr = tr.rec_track
                npts = len(tr.rec_track)
                not_flagged = range(0,npts)
                rec_seg = tr.rec_seg
                rec_swim = tr.rec_swim
            if npts > 1:
                df_track = pd.DataFrame.from_records(rec_tr)
                df_seg = pd.DataFrame.from_records(rec_seg)
                df_swim = pd.DataFrame.from_records(rec_swim)
                if add_smooth:
                    df_seg = df_seg.assign(us = tr.rec_smooth['us'])
                    df_seg = df_seg.assign(vs = tr.rec_smooth['vs'])
                    df_seg = df_seg.assign(nsmooth = tr.rec_smooth['nsmooth'])
                if nt == 0:
                    self.df_all_tracks = df_track
                    self.df_all_segs = df_seg
                    self.df_all_swim = df_swim
                else:
                    self.df_all_tracks = pd.concat([self.df_all_tracks, 
                                                    df_track])
                    self.df_all_segs = pd.concat([self.df_all_segs, df_seg])
                    self.df_all_swim = pd.concat([self.df_all_swim, df_swim])
        if fnames is None:
            fnames.append('tracks.csv')
            fnames.append('segments.csv')
            fnames.append('swim.csv')
        self.df_all_tracks.to_csv(fnames[0], index=False)
        self.df_all_segs.to_csv(fnames[1], index=False)
        self.df_all_swim.to_csv(fnames[2], index=False)

        return

    def plot_study_area(self, fig_name='study_area.png'):
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.plot_bathy(ax)
        self.show_receivers(ax)
        self.format_map(ax)
        plt.savefig(fig_name, dpi=800)

        return

    def plot_pos_single_track(self, variable='position', 
                              color_by='age', plot_noise=False, 
                              plot_smoothed=False, plot_outliers=False,
                              show_receivers=True, plot_masked_only=False,
                              plot_changepts=False, 
                              plot_changepts_fill=False,
                              tracks=None,
                              show_metrics=True, fig_dir=None):
        """ plot information related to each single track/tag """
        if tracks == None:
            tracks = self.tracks.keys()
        nmet = -1 # increment only for "valid" tracks
        for nt, key in enumerate(tracks):
            tr = self.tracks[key]
            if not tr.valid:
                continue
            else:
                nmet += 1
            plt.close()
            fig = plt.figure()
            if plot_noise:
                gs = plt.GridSpec(4,4)
                ax = fig.add_subplot(gs[:-1,:])
            else:
                ax = fig.add_subplot(111)
            self.plot_bathy(ax)
            if show_receivers:
                self.show_receivers(ax)
            if plot_masked_only: 
                mask = tr.get_mask(methods=self.outlier_methods)
            else:
                mask = None
            if 'Cross' in outlier_methods:
                n = self.outlier_methods.index('Cross')
                max_del_vel = self.outlier_params[n][0]
            elif 'Iterative' in outlier_methods:
                n = self.outlier_methods.index('Iterative')
                max_del_vel = self.outlier_params[n][0]
            else:
                max_del_vel = 1.0 # default
            tr.plot_detects(ax, color_by=color_by, 
                            add_cbar=True, 
                            plot_smoothed=plot_smoothed, 
                            plot_outliers=plot_outliers, 
                            plot_changepts=plot_changepts, 
                            plot_changepts_fill=plot_changepts_fill, 
                            max_del_vel=max_del_vel,
                            mask=mask)
            self.format_map(ax)
            if show_metrics:
                self.show_metrics(ax, nmet, variable='position')
                nreds = len(self.reds)
                nyellows = len(self.yellows)
                if nreds > 0:
                    rstring = ''
                    for nr, red in enumerate(self.reds):
                        rstring = rstring + red
                        if nr < nreds - 1:
                            rstring = rstring + ', '
                    ax.set_title('%s: %s'%(key, rstring), color='r')
                elif nyellows > 0:
                    rstring = ''
                    for nr, yellow in enumerate(self.yellows):
                        rstring = rstring + yellow
                        if nr < nyellows - 1:
                            rstring = rstring + ', '
                    ax.set_title('%s: %s'%(key, rstring), color='yellow')
                else:
                    ax.set_title('%s: Good'%(key), color='g')

            # time series part
            if plot_noise:
                ax2 = fig.add_subplot(gs[-1,:])
                tr.plot_noise_distribution(ax2, mask=mask)
                ax2.yaxis.grid(True)

            # save figure
            if fig_dir == None:
                fig_dir = self.default_fig_dir
            os.path.exists(fig_dir) or os.makedirs(fig_dir)
            fig_name = os.path.join(fig_dir, '%s_pos.png'%(key))
            plt.savefig(fig_name, dpi=800)

        return
        
    def plot_bathy(self, ax):
        """ plot bathymetry on grid as background layer """
        self.s_coll = self.grd.plot_cells(values=-self.grd.cells['depth'],
                                          cmap=cm_bath,edgecolor='none',
                                          ax=ax)
        self.s_coll.set_clim([-12,4])

        cticks=np.arange(-12,4.1,4)
        self.cbar = plt.colorbar(self.s_coll, 
                                 ticks=cticks,
                                 label='Bed Elevation (m NAVD)')

        return

    def plot_swim_uv(self, tracks=None):
        """ plot swimming speed histograms """
        plt.close()
        if tracks is None:
            tracks = self.tracks.keys()
        swim_spd = np.array([])
        swim_u = np.array([])
        swim_v = np.array([])
        for nt, key in enumerate(tracks):
            tr = self.tracks[key]
            rec_swim = tr.rec_swim
            swim_spd = np.concatenate((swim_spd,rec_swim.swim_spd))
            swim_u = np.concatenate((swim_u,rec_swim.swim_u))
            swim_v = np.concatenate((swim_v,rec_swim.swim_v))

        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        plt.subplots_adjust(hspace=0.8)
        ax1.hist(swim_spd, bins=50, normed=False, facecolor='g')
        ax1.set_xlabel('swim speed (m s$^{-1}$)')
        ax1.set_xlim([0,0.5])
        ax1.set_ylim([0.0,400])
        ax2 = fig.add_subplot(312)
        ax2.hist(swim_u, bins=50, normed=False, facecolor='g')
        ax2.set_xlabel('streamwise swimming (m s$^{-1}$)')
        ax2.set_xlim([-0.5,0.5])
        ax2.set_ylim([0.0,400])
        ax3 = fig.add_subplot(313)
        ax3.hist(swim_v, bins=50, normed=False, facecolor='g')
        ax3.set_xlabel('lateral swimming (m s$^{-1}$)')
        ax3.set_xlim([-0.5,0.5])
        ax3.set_ylim([0.0,400])

        fig_name = os.path.join(fig_dir, 'swim_hist.png')
        plt.savefig(fig_name, dpi=800)

        return

    def plot_swim_uv_daynight(self, tracks=None):
        """ plot swimming speed histograms for 
            daytime and nightime entry to array"""
        plt.close()
        if tracks is None:
            tracks = self.tracks.keys()
        swim_spd_d = np.array([])
        swim_u_d = np.array([])
        swim_v_d = np.array([])
        swim_spd_n = np.array([])
        swim_u_n = np.array([])
        swim_v_n = np.array([])
        for nt, key in enumerate(tracks):
            tr = self.tracks[key]
            rec_swim = tr.rec_swim
            i = tr.mnames.index('daytime_entry')
            day = tr.metrics[i]
            if day:
                swim_spd_d = np.concatenate((swim_spd_n,rec_swim.swim_spd))
                swim_u_d = np.concatenate((swim_u_n,rec_swim.swim_u))
                swim_v_d = np.concatenate((swim_v_n,rec_swim.swim_v))
            else:
                swim_spd_n = np.concatenate((swim_spd_n,rec_swim.swim_spd))
                swim_u_n = np.concatenate((swim_u_n,rec_swim.swim_u))
                swim_v_n = np.concatenate((swim_v_n,rec_swim.swim_v))

        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        plt.subplots_adjust(hspace=0.8)
        ax1.hist(swim_spd_d, bins=np.arange(0.0,0.501,0.025), normed=True, 
                 edgecolor='r', facecolor='None', label='day')
        ax1.hist(swim_spd_n, bins=np.arange(0.0,0.501,0.025), normed=True, 
                 edgecolor='k', facecolor='None', label='night')
        ax1.legend(loc='right')
        ax1.set_xlabel('swim speed (m s$^{-1}$)')
        ax1.set_xlim([0,0.5])
#       ax1.set_ylim([0.0,400])
        ax2 = fig.add_subplot(312)
        ax2.hist(swim_u_d, bins=np.arange(-0.5,0.5,0.05), normed=True,
                 edgecolor='r', facecolor='None')
        ax2.hist(swim_u_n, bins=np.arange(-0.5,0.5,0.05), normed=True, 
                 edgecolor='k', facecolor='None')
        ax2.set_xlabel('streamwise swimming (m s$^{-1}$)')
        ax2.set_xlim([-0.5,0.5])
#       ax2.set_ylim([0.0,400])
        ax3 = fig.add_subplot(313)
        ax3.hist(swim_v_d, bins=np.arange(-0.5,0.5,0.05), normed=True, 
                 edgecolor='r', facecolor='None')
        ax3.hist(swim_v_n, bins=np.arange(-0.5,0.5,0.05), normed=True, 
                 edgecolor='k', facecolor='None')
        ax3.set_xlabel('lateral swimming (m s$^{-1}$)')
        ax3.set_xlim([-0.5,0.5])
#       ax3.set_ylim([0.0,400])

        fig_name = os.path.join(fig_dir, 'swim_hist_daynight.png')
        plt.savefig(fig_name, dpi=800)

        return

    def plot_transit_times_daynight(self, tracks=None):
        """ plot transit time histograms for 
            daytime and nightime entry to array"""
        plt.close()
        if tracks is None:
            tracks = self.tracks.keys()
        time_d = []
        time_n = []
        for nt, key in enumerate(tracks):
            tr = self.tracks[key]
            rec_swim = tr.rec_swim
            i = tr.mnames.index('daytime_entry')
            day = tr.metrics[i]
            transit_time = rec_swim['dn2'][-1] - rec_swim['dn1'][0]
            transit_time = transit_time*24.*60. # convert to minutes
            if day:
                time_d.append(transit_time)
            else:
                time_n.append(transit_time)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.hist(time_d, bins=np.arange(0.0,30.,3.), normed=False, 
                 edgecolor='r', facecolor='None', label='day')
        ax1.hist(time_n, bins=np.arange(0.0,30.,3.), normed=False, 
                 edgecolor='k', facecolor='None', label='night')
        ax1.legend(loc='right')
        ax1.set_xlabel('transit time (minutes)')
        ax1.set_xlim([0,30])
#       ax1.set_ylim([0.0,400])

        fig_name = os.path.join(fig_dir, 'transit_time.png')
        plt.savefig(fig_name, dpi=800)

        return

    def get_transit_times(self, tracks=None):
        """ calculate transit times """
        if tracks is None:
            tracks = self.tracks.keys()
        ttimes = []
        for nt, key in enumerate(tracks):
            tr = self.tracks[key]
            rec = tr.rec_smooth_pos
            transit_time = rec['dnums'][-1] - rec['dnums'][0]
            transit_time = transit_time*24.*60. # convert to minutes
            ttimes.append(transit_time)

        return ttimes

    def plot_transit_times_z(self, tracks=None):
        """ plot transit time histograms against flow """
        plt.close()
        if tracks is None:
            tracks = self.tracks.keys()
        transit_times = self.get_transit_times(tracks)
        q_transit = []
        avg_zs = []
        for nt, key in enumerate(tracks):
            tr = self.tracks[key]
            rec= tr.rec_track
            avg_z= np.average(rec.Z)
            avg_zs.append(avg_z)
            
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(avg_zs, transit_times,'*')
        ax1.set_xlabel('Z (m)')
        ax1.set_ylabel('Transit Time (minutes)')
        ax1.set_xlim([0,10])
        ax1.set_ylim([0,60])
        fig_name = os.path.join(fig_dir, 'transit_time_z.png')
        plt.ion()
        plt.show()
        pdb.set_trace()
        plt.savefig(fig_name, dpi=800)

        return

    def plot_transit_times_flow(self, dnums_q, q, tracks=None):
        """ plot transit time histograms against flow """
        plt.close()
        if tracks is None:
            tracks = self.tracks.keys()
        transit_times = self.get_transit_times(tracks)
        q_transit = []
        midpoint_times = []
        for nt, key in enumerate(tracks):
            tr = self.tracks[key]
            rec= tr.rec_smooth_pos
            midpoint_time = 0.5*(rec['dnums'][-1] + rec['dnums'][0])
            midpoint_times.append(midpoint_time)
            
        q_interp = interp1d(dnums_q, q)
        q_midpoint = q_interp(midpoint_times)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(q_midpoint, transit_times,'*')
        ax1.set_xlabel('Mossdale Flow (m$^3$s$^{-1}$)')
        ax1.set_ylabel('Transit Time (minutes)')
        ax1.set_xlim([0,300])
        ax1.set_ylim([0,60])
        fig_name = os.path.join(fig_dir, 'transit_time_flow.png')
        plt.ion()
        plt.show()
        plt.savefig(fig_name, dpi=800)

        return

    def calc_crossings(self, lines, tracks=None):
        """ plot cumulative number of crossings from river left """
        plt.close()
        if tracks is None:
            tracks = self.tracks.keys()
        
        fraction = {}
        sort_frac = {}
        for lkey in lines.keys():
            fraction[lkey] = []
            b1 = lines[lkey][0]
            b2 = lines[lkey][1]
            for nt, key in enumerate(tracks):
                tr = self.tracks[key]
                not_flagged = np.where(tr.rec_track.flagged==0)[0]
                rec_track = tr.rec_track[not_flagged]
                # rec_seg might already be set
                # want to be sure it uses only valid points
                rec_seg = tr.make_segments(set_depth=True, 
                                           input_rec_track=rec_track)
                for ns, seg in enumerate(rec_seg):
                    a1 = [seg.x1, seg.y1]
                    a2 = [seg.x2, seg.y2]
                    xy = get_intersect(a1,a2,b1,b2)
                    if b1[0] < xy[0] < b2[0]:
                        frac = (xy[0] - b1[0])/(b2[0] - b1[0])
                        print( "lkey, key, ns, frac",lkey,key,ns,frac)
                        fraction[lkey].append(frac)

            ncross = len(fraction[lkey])
            sort_frac[lkey] = np.sort(fraction[lkey])
            plt.step(sort_frac[lkey], np.arange(ncross), label=lkey)
        plt.gca().legend(loc='upper left')
        plt.gca().set_xlabel('Lateral Position')
        plt.gca().set_ylabel('Cumulative Detections')
        fig_name = os.path.join(fig_dir, 'lateral_distribution.png')
        plt.savefig(fig_name, dpi=800)
        return

    def plot_cell_detects(self, normalize=True, masked=False, tracks=None):
        """ plot number of detects in each grid cell, 
            optionally normalized by cell area """
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ncells = len(self.grd.cells['depth'])
        detects_i = np.zeros(ncells, np.int32)
        detects_plot = np.zeros(ncells, np.float64)
        if tracks is None:
            tracks = self.tracks.keys()
        for nt, key in enumerate(tracks):
            tr = self.tracks[key]
            detects_i_tr = tr.cell_detects(masked=masked)
            detects_i += detects_i_tr
        if normalize:
            detects_plot = np.divide(detects_i, self.grd.cells['_area'])
        else:
            detects_plot = detects_i
        self.s_coll = self.grd.plot_cells(values=detects_plot,
                                          cmap=cmap_r,edgecolor='none', ax=ax)
        self.s_coll.set_clim([0,5])
        self.cbar = plt.colorbar(self.s_coll, label='Detections')
        self.show_receivers(ax, color='k')
        self.format_map(ax)
        fig_name = os.path.join(fig_dir, 'detects_in_cell.png')
        plt.savefig(fig_name, dpi=800)

    def plot_cell_speed(self):
        """ plot map showing average of all speed estimates in each cell """
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ncells = len(self.grd.cells['depth'])
        detects_i = np.zeros(ncells, np.int32)
        speed_sum_i = np.zeros(ncells, np.float64)
        for nt, key in enumerate(self.tracks.keys()):
            tr = self.tracks[key]
            detects_i_tr, speed_sum_i_tr = tr.cell_speed()
            detects_i += detects_i_tr
            speed_sum_i += speed_sum_i_tr
        speed_i = np.divide(speed_sum_i, detects_i)
        self.s_coll = self.grd.plot_cells(values=speed_i,
                                          cmap=cmap_r,edgecolor='none', ax=ax)
        self.s_coll.set_clim([0,2])
        self.cbar = plt.colorbar(self.s_coll, label='Speed (m s$^{-1}$)')
        self.show_receivers(ax, color='k')
        self.format_map(ax)
        fig_name = os.path.join(fig_dir, 'speed_in_cell.png')
        plt.savefig(fig_name, dpi=800)

    def plot_noise_histograms(self, tracks):
        """ plot histogram of difference between position estimate
            and smoothed position estimate """
        plt.close()
        fig = plt.figure()
        x, y, noise, node_ct = calc_noise(tracks)
        for nc in range(3,6):
            subplot_num = 310 + (nc-2)
            ax = fig.add_subplot(subplot_num)
            if nc < 5:
                valid = np.where(node_ct == nc)[0]
            if nc >= 5:
                valid = np.where(node_ct >= nc)[0]
            ax.hist(noise[valid], bins=50, normed=False, facecolor='g')
            ax.set_xlabel('position error (m)')
            ax.set_xlim([0,4])
            ax.text(0.75, 0.86, "n = %d"%len(valid), transform=ax.transAxes)
            ax.text(0.75, 0.72, "max = %4.2f"%np.max(noise[valid]), transform=ax.transAxes)
            ax.text(0.75, 0.58, "avg = %4.2f"%np.average(noise[valid]), transform=ax.transAxes)
            ax.text(0.75, 0.44, "nodes = %2d"%nc, transform=ax.transAxes)
             
        fig_name = os.path.join(fig_dir, 'position_noise.png')
        plt.savefig(fig_name, dpi=800)

    def calc_noise(self, tracks):
        """ calculate estimate of position noise """
        noise = []
        node_ct = []
        x = []
        y = []
        for nt, key in enumerate(tracks):
            tr = self.tracks[key]
            not_flagged = np.where(tr.rec_track.flagged==0)[0]
            rec_tr = tr.rec_track[not_flagged]
            rec_sm = tr.rec_smooth_pos
            nct = rec_tr.Node_count
            xdiff = rec_tr.X - rec_sm.X
            ydiff = rec_tr.Y - rec_sm.X
            xydiff = np.sqrt(np.power(xdiff,2) + np.power(ydiff,2))
            valid = np.where(np.isfinite(xydiff))[0]
            x += rec_tr.X[valid].tolist()
            y += rec_tr.Y[valid].tolist()
            node_ct += nct[valid].tolist()
            noise += xydiff[valid].tolist()
        node_ct = np.asarray(node_ct)
        noise = np.asarray(noise)
        x = np.asarray(x)
        y = np.asarray(y)

        return x, y, node_ct, noise

    def plot_noise_map(self, masked=True, tracks=None):
        """ show individual position noise estimates on map """
        x, y, noise, node_ct = self.calc_noise(tracks)
        noise_max=10.
        for nc in range(3,7):
            plt.close()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if nc < 5:
                valid = np.where(node_ct == nc)[0]
            elif nc == 5:
                valid = np.where(node_ct >= nc)[0]
            elif nc > 5:
                valid = np.where(node_ct >= 1)[0]
            self.plot_bathy(ax)
            self.show_receivers(ax, color='k')
            self.format_map(ax)
            noi = ax.scatter(x[valid], y[valid], 
                             c=cm_red(noise[valid]/noise_max), 
                             marker='.',s=4.0)
            cdata = np.arange(noise_max, 0, -.1).reshape(10, 10)
            cax = fig.add_axes([0.52,0.63,0.20,0.025])
            im = ax.imshow(cdata, cmap=cm_red)
            cbar_noi = fig.colorbar(im, cax=cax, orientation='horizontal')
            cbar_noi.set_label('Position Noise (m)')
            ax.text(0.03, 0.24, "n = %d"%len(valid), transform=ax.transAxes)
            ax.text(0.03, 0.18, "max = %4.2f"%np.max(noise[valid]), transform=ax.transAxes)
            ax.text(0.03, 0.12, "avg = %4.2f"%np.average(noise[valid]), transform=ax.transAxes)
            ax.text(0.03, 0.06, "nodes = %2d"%nc, transform=ax.transAxes)
             
            fig_name = 'position_noise_%d.png'%(nc)
            fig_name = os.path.join(fig_dir, fig_name)
            plt.savefig(fig_name, dpi=800)

        return

    def plot_non_detects(self, normalize=True, masked=False, tracks=None):
        """ plot map of density of missing detections in each grid cell """
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ncells = len(self.grd.cells['depth'])
        non_detects_i = np.zeros(ncells, np.int32)
        if tracks is None:
            tracks = self.tracks.keys()
        for nt, key in enumerate(tracks):
            tr = self.tracks[key]
            non_detects_i_tr = tr.nondetects(masked=masked)
            non_detects_i += non_detects_i_tr
        if normalize:
            ndetects_plot = np.divide(non_detects_i, self.grd.cells['_area'])
        else:
            ndetects_plot = non_detects_i
        self.s_coll = self.grd.plot_cells(values=ndetects_plot,
                                          cmap=cmap_r,edgecolor='none', ax=ax)
        self.s_coll.set_clim([0,5])
        self.cbar = plt.colorbar(self.s_coll, label='Non Detects')
        self.show_receivers(ax, color='k')
        self.format_map(ax)
        fig_name = os.path.join(fig_dir, 'non_detects.png')
        plt.savefig(fig_name, dpi=800)

        return

    def plot_occupancy(self, normalize=True, masked=False, tracks=None):
        """ plot map of number of unique tag detectsion in each cell,
            optionally normalized by cell area """
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ncells = len(self.grd.cells['depth'])
        occupancy_i = np.zeros(ncells, np.int32)
        if tracks is None:
            tracks = self.tracks.keys()
        for nt, key in enumerate(tracks): # "tracks" is set of keys
            tr = self.tracks[key]         # self.tracks are the track objects
            i_list = tr.cell_occupancy(masked=masked)
            occupancy_i[i_list] += 1
        if normalize:
            occupancy_plot = np.divide(occupancy_i, self.grd.cells['_area'])
        else:
            occupancy_plot = occupancy_i
        self.s_coll = self.grd.plot_cells(values=occupancy_plot,
                                          cmap=cmap_r,edgecolor='none', ax=ax)
        self.s_coll.set_clim([0,1])
        self.cbar = plt.colorbar(self.s_coll, label='Occupancy')
        self.show_receivers(ax, color='k')
        self.format_map(ax)
        fig_name = os.path.join(fig_dir, 'occupancy.png')
        plt.savefig(fig_name, dpi=800)

        return

    def calculate_quality_metrics(self, masked=False, tracks=None):
        """ calculate metrics of overall track, as opposed to 
            individual detection, quality """
        metrics = defaultdict(list) # {}
        if tracks is None:
            tracks = self.tracks.keys()
        for nt, key in enumerate(tracks):
            tr = self.tracks[key]
            if not tr.valid:
                continue
            if masked:
                # check if "flagged" is set
                if 'flagged' not in tr.rec_track.dtype.names:
                    tr.total_outlier_flag()
            mnames, metrics_tr = tr.track_quality_metrics(masked=masked)
            for nn, mname in enumerate(mnames):
                metrics[mname].append(metrics_tr[nn])

        df_metrics = pd.DataFrame.from_dict(metrics)
        metrics_csv = os.path.join(self.csv_dir, 'metrics.csv')
        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)
        df_metrics.to_csv(metrics_csv, index=False)
        self.df_metrics = df_metrics 

        return

    def classify_by_quality(self):
        """ the goal of this is to define trusted tracks purely on the 
            basis of objective criteria """
        df = self.df_metrics
        IDs = df['ID'].values
        poor_quality = np.zeros(self.ntracks, np.int32)
        quality_dict = {}
        exclude_tracks = []
        for nt, key in enumerate(self.tracks.keys()):
            tr = self.tracks[key]
            quality_dict[key] = '|'
            if not tr.valid:
                exclude_tracks.append(key)
                quality_dict[key] += 'invalid|'
            else:
                row = np.where(IDs == key)[0]
                frac_detects = df['frac_detects'].values[row]
                vel_sd = df['vel_sd'].values[row]
                quality_dict[key] = '|'
                if frac_detects < 0.1:
                    poor_quality[nt] = 1 # 1 = True
                    quality_dict[key] += 'nondetects|'
                if vel_sd > 0.6:
                    poor_quality[nt] = 1 # 1 = True
                    quality_dict[key] += 'vel_sd|'
                #nvalid = sum(tr.rec_track.flagged == 0)
                nvalid = len(tr.rec_smooth)
                #dnums = tr.rec_track.dnums
                #transit_time = dnums[-1] - dnums[0]
                #transit_time = transit_time*24. # convert to hours
                t_sec = tr.rec_smooth['tm'][-1] - tr.rec_smooth['tm'][0]
                t_hours = t_sec/3600.
                if (nvalid < 25 or t_hours > 1.):
                    exclude_tracks.append(key)
#               if poor_quality[nt] > 0:
#                   exclude_tracks.append(key)
        self.exclude_tracks = exclude_tracks
        # append to df_metrics?

        return

    def add_vel_scale(self, ax, tr):
        """ show velocity scale on map figure """
        qxy = [0.20, 0.25]
        key_scale = 0.5 # 0.05
        qk_label = '%3.1f m s$^{-1}$'%key_scale
        self.qk = ax.quiverkey(tr.quiv, qxy[0], qxy[1], key_scale, qk_label)

        return

    def show_receivers(self, ax, color='k'):
        """ shows location of acoustic receivers on map """
        rcv = self.rcv_rec
        ax.scatter(rcv.X, rcv.Y, marker='+', c=color, s=5.)

    def format_map(self, ax):
        """ common formatting used on multiple map figuresp """
        ax.axis('scaled')
        ax.set_xlim(xylims[0])
        ax.set_ylim(xylims[1])
        show_axis = False
        if show_axis:
            ax.set_xlabel('Easting (m)')
            ax.set_ylabel('Northing (m)')
        else:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.axis('off')
        #ax.autoscale(tight=True)

    def show_metrics(self, ax, nt, variable='speed'):
        """ show track quality metrics on a map """
        key = self.tracks.keys()[nt]
        self.yellows = []
        self.reds = []
        t0xy = [0.00, 0.17]
        m0 = self.df_metrics['t_elapsed'].values[nt]/3600.
        t0 = "hours elapsed: %4.2f"%m0
        c0 = 'k' # default
        if m0 > 4.0:
            c0 = 'r'
            self.reds.append('long occupancy')
        elif m0 > 1.0:
            c0 = 'yellow'
            self.yellows.append('long occupancy')
        ax.text(t0xy[0], t0xy[1], t0, transform=ax.transAxes, color=c0)
        t1xy = [0.00, 0.09]
        m1 = self.df_metrics['frac_detects'].values[nt]
        t1 = "fraction detects: %4.2f"%m1
        c1 = 'k' # default
        if m1 < 0.2:
            c1 = 'r'
            self.reds.append('low detects')
        elif m1 < 0.4:
            c1 = 'yellow'
            self.yellows.append('low detects')
        ax.text(t1xy[0], t1xy[1], t1, transform=ax.transAxes, color=c1)
        t2xy = [0.00, 0.01]
        c2 = 'k' # default
        if variable == 'speed':
            m2 = self.df_metrics['dvel_mag'].values[nt]
            t2 = "velocity variability: %4.2f m s$^{-1}$"%m2
            if m2 > 0.8:
                c2 = 'r'
                self.reds.append('noisy')
            elif m2 > 0.6:
                c2 = 'yellow'
                self.yellows.append('noisy')
        elif variable == 'position':
            m2 = self.df_metrics['pos_error'].values[nt]
            t2 = "position error: %4.2f m"%m2
            if m2 > 4.0:
                c2 = 'r'
                self.reds.append('noisy')
            elif m2 > 2.0:
                c2 = 'yellow'
                self.yellows.append('noisy')
        ax.text(t2xy[0], t2xy[1], t2, transform=ax.transAxes, color=c2)


if True: # __name__ == '__main__':
    # avoid popping up figures as they're generated
    with plt.rc_context(rc={'interactive': False}):
        #lines_shp_file = r'R:\UCD\Projects\CDFW_Klimley\GIS\transit_lines.shp'
        #lines = get_lines_from_shp(lines_shp_file)
        if 0:
            flow_fn = 'bc_flows.csv'
            dtime_msd, q_msd = load_comma_formatted(flow_fn,'MSD')
            dnums_msd = pylab.date2num(dtime_msd)
        else:
            local_csv='msd_2018.csv'
            df=pd.read_csv('msd_flow_2018.csv')
            q_msd=df['q_cms'].values
            dnums_msd=df['dnum'].values

        filename = 'cleaned_half_meter.csv'
        # filename = r'R:\UCD\Projects\CDFW_Klimley\Observations\telemetry\8294.csv'
        grd_file = 'FISH_PTM.grd'
        shp_file = 'receiver_range_polygon.shp'
        fig_dir = 'figs_trusted'
        csv_dir = 'csv'
        # This is what was used in figures for Ed and Mike's presentations
        # I believe it's from a 2D suntans run.
        #hydro_fname = '../../field/tags/segments_2m-model20190314.csv'
        # This has the most recent hydro, and a more recent segments input
        # than 20190314, but the results don't look as good. maybe because
        # of using model_u, instead of model_u_surf.  trying now with
        # track.py changed to use model_{u,v}_surf
        hydro_fname = '../../../field/tags/segments_m-model20191202.csv'
        # This is somewhere in between.
        #hydro_fname = '../../../field/tags/segments_m-model20190710.csv'
        outlier_methods = ['Poly','Dry','Iterative']
        #outlier_params = [[sur.names,sur.polys],[-3.0],[2.0]]

        os.path.exists(fig_dir) or os.makedirs(fig_dir)

        autocorr = False
        if autocorr:
            trusted_tracks_ac=['7629']
            sur = Survey(filename, grd_file, shp_file, trackIDs=trusted_tracks_ac) 
            sur.speed_over_ground(masked=False)
            outlier_params = [[sur.names,sur.polys],[-3.0],[2.0]]
            sur.identify_outliers(outlier_methods, outlier_params)
            sur.calc_autocorr(masked=True)
            raise Exception("Bailing out") # sys.exit(0) # hardwire for now
        debug = False
        if debug:
            # debug first clean point in track 8294
            trusted_tracks=['8294']
            sur = Survey(filename, grd_file, shp_file, trackIDs=trusted_tracks) 
            sur.set_directories(default_fig_dir=fig_dir, csv_dir=csv_dir)
            sur.read_receiver_locations(fname='2018 surveyed_rcv_pos.csv')
            sur.speed_over_ground(masked=False)
            outlier_params = [[sur.names,sur.polys],[-3.0],[2.0]]
            print( "call identify outliers")
            sur.identify_outliers(outlier_methods, outlier_params)
            sur.calc_smoothed(masked=True)
            sur.swimming_speed(hydro_fname, tracks=trusted_tracks)
            tag=trusted_tracks[0]
            sur.write_to_csvs(fnames=['tag%s.csv'%tag,
                                      'tag%s_seg.csv'%tag,
                                      'tag%s_swim.csv'%tag],
                              add_smooth=False)
            raise Exception("Debug output enabled - bailing out")

        # screen tracks
        process_all_tracks = False 
        if process_all_tracks:
            sur = Survey(filename, grd_file, shp_file) # read survey ALL TRACKS
            sur.set_directories(default_fig_dir=fig_dir, csv_dir=csv_dir)
            sur.read_receiver_locations(fname='2018 surveyed_rcv_pos.csv')
            sur.speed_over_ground(masked=False)
            outlier_params = [[sur.names,sur.polys],[-3.0],[2.0]]
            sur.identify_outliers(outlier_methods, outlier_params)
            # plot raw tracks
            plot_raw = False
            if plot_raw:
                sur.plot_pos_single_track(variable='position', color_by='age',
                                          plot_noise=False, plot_smoothed=False,
                                          plot_outliers=False, plot_masked_only=False,
                                          plot_changepts=False, show_metrics=False,
                                          fig_dir='ALL_raw_tracks')
            # first do screening of track quality for ALL TRACKS
            plot_outliers = True
            if plot_outliers:
                sur.calc_smoothed(masked=True)
                sur.calculate_quality_metrics(masked=True)
                sur.classify_by_quality()
    #           sur.plot_pos_single_track(variable='position', color_by='age',
    #                                     plot_noise=False, plot_smoothed=False,
    #                                     plot_outliers=True, plot_masked_only=False,
    #                                     plot_changepts=False, show_metrics=True,
    #                                     fig_dir='ALL_plot_outliers')
                trusted_new = [tag for tag in sur.tracks.keys() 
                               if tag not in sur.exclude_tracks]
            plot_cleaned = False
            if plot_cleaned:
                sur.plot_pos_single_track(variable='position', color_by='age',
                                          plot_noise=False, plot_smoothed=False,
                                          plot_outliers=False, plot_masked_only=True,
                                          plot_changepts=False, show_metrics=False,
                                          fig_dir='plot_cleaned')


        # From here down restart to work with "trusted tracks"
        # manualy selected tracks
        #   trusted_tracks=['7265','72d1','74a1','74ab','7528','752b','7566','7567','7629','7659','768a','76b1','76b7','772a','7895','796d','79ab','7a85','7a96','7a99','7b49','7b4b','7b65','7ba9','7ca5','7d29','7dd5','8255','8292','7435']
        # extracted from unique ids in segments_2m-model20190314.csv
        trusted_tracks=['8255', '7dd5', '7b4b', '796d', '74ab', '8292', '72d1', '7659',
                        '752b', '7ba9', '768a', '7265', '772a', '7a85', '7d29', '7435',
                        '74a1', '7a99', '7b65', '7629', '7b49', '7566', '7567', '76b7',
                        '76b1', '7a96', '7ca5', '79ab', '7528', '7895']
        # Just the ones used in Mike's ppt:
        trusted_tracks=['7567','772a', '8255', '7b65',  '8292', '7a96']
        # screening based on fraction detects 
        #   trusted_tracks = ['8255', '7c55', '796d', '75d3', '75d5', '7af5', '752b', '7454', '768a', '7ca5', '7615', '7a85', '7d65', '76da', '7b49', '725b', '7cb5', '7a99', '7d55', '76d3', '7256', '769a', '76d9', '7a96', '7a95', '7522', '755b', '79ab', '7528', '7895', '82a4', '7b2b', '755e', '7975', '78ab', '754f', '7435', '77a5', '82d5', '7629', '7bd5', '7599', '7275', '7492', '772a', '7b4d', '7b4b', '8292', '74ca', '74cb', '7659', '7d25', '7499', '7269', '74dd', '7265', '75b6', '7d29', '7aa3', '76a7', '7555', '734d', '74c9', '7a51', '7566', '7567', '76ad', '76ae', '76af', '7692', '7ab2', '74b5', '7289', '7ad7', '74ab', '7d49', '72d1', '7aed', '7ba9', '7577', '76bd', '758a', '74a1', '75c9', '7ae9', '7a6a', '836a', '75ba', '8294', '728d', '7b65', '7adb', '82ba', '76b7', '76b1', '76cb', '7585', '7acd']
        # trusted_tracks = ['8255', '7c55', '796d', '75d5', '752b', '7454', '768a', '7ca5', '7615', '7a85', '7b49',
        #                   '7cb5', '7a99', '7d55', '76d9', '7a96', '7a95', '7528', '7895', '7b2b', '7435', '77a5',
        #                   '82d5', '7629', '7bd5', '7275', '7b4d', '8292', '74ca', '7659', '7499', '7265', '7d29',
        #                   '734d', '7a51', '7566', '7567', '76af', '7ab2', '7ad7', '74ab', '7d49', '72d1', '7577',
        #                   '74a1', '7ae9', '7a6a', '7b65', '7adb', '76b7', '76b1', '7acd']

        # RH - debugging track 7c55, which is supposed to be trusted, yet it has no segments.
        # trusted_tracks=['7c55']
        # screening: ndetects > 25, t_transit < 1 hour

        sur = Survey(filename, grd_file, shp_file, trackIDs=trusted_tracks) # read survey
        sur.set_directories(default_fig_dir=fig_dir, csv_dir=csv_dir)
        sur.read_receiver_locations(fname='2018 surveyed_rcv_pos.csv')
        sur.speed_over_ground(masked=False)
        outlier_params = [[sur.names,sur.polys],[-3.0],[2.0]]
        sur.identify_outliers(outlier_methods, outlier_params)
        sur.calc_smoothed(masked=True)
        plot_cleaned = True
        if plot_cleaned:
            print( "plot cleaned")
            sur.plot_pos_single_track(variable='position', color_by='age',
                                      plot_noise=False, plot_smoothed=False,
                                      plot_outliers=False, plot_masked_only=True,
                                      plot_changepts=False, show_metrics=False,
                                      fig_dir='plot_cleaned_trusted')
        plot_smoothed = True
        if plot_smoothed:
            print( "plot smoothed")
            sur.plot_pos_single_track(variable='position', color_by='age',
                                      plot_noise=False, plot_smoothed=True,
                                      plot_outliers=False, plot_masked_only=True,
                                      plot_changepts=False, show_metrics=False,
                                      fig_dir='plot_smoothed_trusted')
        sur.find_change_points(tracks=trusted_tracks)
        sur.find_change_points_fill(tracks=trusted_tracks)
        plot_changepts = False
        if plot_changepts:
            print( "plot changepoints")
            sur.plot_pos_single_track(variable='position', color_by='age',
                                      plot_noise=False, plot_smoothed=True,
                                      plot_outliers=False, plot_masked_only=True,
                                      plot_changepts=True, show_metrics=False,
                                      tracks=trusted_tracks,
                                      fig_dir='plot_changepts_trusted')
        plot_changepts_fill = False
        if plot_changepts_fill:
            print( "plot changepoints_fill")
            sur.plot_pos_single_track(variable='position', color_by='age',
                                      plot_noise=False, plot_smoothed=True,
                                      plot_outliers=False, plot_masked_only=True,
                                      plot_changepts=False, 
                                      plot_changepts_fill=True, 
                                      show_metrics=False,
                                      tracks=trusted_tracks,
                                      fig_dir='plot_changepts_fill_trusted')
        sur.speed_over_ground(masked=True)
        plot_speed = True
        if plot_speed:
            print( "plot speed over ground")
            sur.plot_speed_single_track(variable='speed_over_ground', 
                                        color_by='age',
                                        show_metrics=False,
                                        time_series=True, 
                                        histograms=False, 
                                        fig_dir='speed_over_ground_trusted',
                                        masked=True,
                                        tracks=trusted_tracks)
        sur.swimming_speed(hydro_fname, tracks=trusted_tracks)
        plot_swim = True
        if plot_swim:
            print( "plot swimming speed")
            sur.plot_speed_single_track(variable='swimming_speed', 
                                        plot_detects=False,
                                        show_metrics=False,
                                        histograms=True,
                                        fig_dir='swim_speed_trusted',
                                        masked=True,
                                        tracks=trusted_tracks)
        plot_hydro = False
        if plot_hydro:
            print( "plot hydro speed")
            sur.plot_speed_single_track(variable='hydro_speed', 
                                        plot_detects=False,
                                        show_metrics=False,
                                        histograms=False,
                                        fig_dir='hydro_speed_trusted',
                                        masked=True,
                                        tracks=trusted_tracks)

        #   sur.calc_crossings(lines, tracks=trusted_tracks)
        #   sur.plot_transit_times_z(tracks=trusted_tracks)
        if dnums_msd is not None:
            sur.plot_transit_times_flow(tracks=trusted_tracks, 
                                        dnums_q=dnums_msd, q=q_msd)
        else:
            print("-"*20 + "No transit time -- missing dnums_msd" + "-"*20)

        output_summary = False
        if output_summary:
            print( "integrated plots")
    #       sur.plot_study_area()
    #       sur.write_to_csvs(fnames=['tracks_2.csv','segments_2.csv'])
            sur.write_to_csvs(fnames=['tracks_2m.csv','segments_2m.csv',
                                      'swim_2m.csv'], masked=True)
            sur.calculate_quality_metrics(masked=True) # need for daytime entry calc
            sur.plot_all_detects(daynight=True, fig_name='all_detects_daynight.png')
            sur.plot_all_detects(routesel=True, fig_name='all_detects_route.png')
            sur.plot_swim_uv(tracks=trusted_tracks)
            sur.plot_swim_uv_daynight(tracks=trusted_tracks)
            sur.plot_transit_times_daynight(tracks=trusted_tracks)
            sur.plot_occupancy(normalize=True, masked=True, tracks=trusted_tracks)
            sur.plot_noise_map(masked=True, tracks=trusted_tracks)
            sur.plot_cell_detects(normalize=True, masked=True, daynight=True,
                                  tracks=trusted_tracks)
            sur.plot_cell_speed()
            sur.plot_non_detects(normalize=True, masked=True, tracks=trusted_tracks)
            sur.plot_noise_scatter(masked=True)
    # change masked to true below. Need to generalize for smoothed points
            sur.plot_noise_histograms(trusted_tracks)
            sur.plot_all_speed_vectors(variable='speed_over_ground')
