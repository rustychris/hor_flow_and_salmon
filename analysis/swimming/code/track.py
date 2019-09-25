# ed
# methods for dealing with a single acoustic telemetry track 
import sys, os 
import numpy as np 
import pandas as pd
from pylab import date2num as d2n
from datetime import datetime
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import stats
import stompy.plot.cmap as cmap
import math
from scipy.stats import circstd, circmean
from shapely import geometry
import copy
import pdb

cm_age = plt.get_cmap('summer')
cm_red = plt.get_cmap('autumn_r')
cm_autumn = plt.get_cmap('autumn')
cm_red_r = plt.get_cmap('Reds_r')

outlier_marker = {'Poly':'o',
                  'Dry':'x',
                  'Tek':'^',
                  'Cross':'>',
                  'Potts':'<',
                  'Iterative':'v'}
outlier_color = {'Poly':'r',
                 'Dry':'r',
                 'Tek':'pink',
                 'Cross':'magenta',
                 'Potts':'salmon',
                 'Iterative':'r'}

dt_signal = 5
age_ticks = [10,100,1000,10000]
ticks = [np.log10(a) for a in age_ticks]
tick_labels = ['$\mathregular{10^%d}$'%a for a in ticks]
qscale = 0.06
depth_min = 0.01

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
    
# to do:
# - pass in DEM
# - pass in elevation time series
# - set depth at xm,ym,tm = max(0.0,eta(tm) - bed_elev(xm,ym))
# - use interpolated velocity to get uh, vh - hydro velocity 
# - difference to get swimming speed

class Track(object):
    """ A track object is associated with data and actions for 
        a single acoustic tag """

    def __init__(self, 
                 df_track = None,
                 grd = None):
        self.df_track = df_track
        self.grd = grd
        # automatically convert to defined type
        self.define_track() # can convert to defined type

    def define_track(self, set_depth=True):
        """ read in and set fields of a single track """ 
        grd = self.grd
        self.ID = self.df_track['ID'].values[0]
        self.valid = True
        self.ndetects = len(self.df_track)
        if self.ndetects < 3:
            self.valid = False
        # save redundant ID information for each detection
        dstrings = self.df_track['dt'].values
        dtimes = [datetime.strptime(ds, '%m/%d/%Y %H:%M') for ds in dstrings]
        dnums = d2n(dtimes)
        self.df_track = self.df_track.assign(dnums = dnums)
        self.t_entry = self.df_track['Sec'].values[0]
        if set_depth:
            depth = np.nan*np.ones(self.ndetects, np.float64)
            idetect = -1*np.ones(self.ndetects, np.int32)
            for nd in range(self.ndetects):
                xy = [self.df_track['X'].values[nd],
                      self.df_track['Y'].values[nd]]
                i = grd.select_cells_nearest(xy, inside=True)
                # seems the interface may have changed -- test
                # both ways
                if (i is not None) and (i>=0):
                    idetect[nd] = i
                    depth[nd] = grd.cells['depth'][i]
            self.df_track = self.df_track.assign(depth = depth)
            self.df_track = self.df_track.assign(i = idetect.astype(np.int32))
        self.df_track = self.df_track.assign(nd = range(self.ndetects))
        self.rec_track = self.df_track.to_records()
        return

    def set_swim(self, swim_dframe=None):
        self.swim_dframe = swim_dframe
        self.rec_swim = swim_dframe.to_records()
        uh = self.rec_swim['model_u']
        vh = self.rec_swim['model_v']
        #swim_u = self.rec_swim['u'] - uh
        #swim_v = self.rec_swim['v'] - vh
        swim_u = self.rec_swim['us'] - uh
        swim_v = self.rec_swim['vs'] - vh
        nsegments = len(self.rec_swim)
        hydro_head = np.zeros(nsegments, np.float64)
        swim_head = np.zeros(nsegments, np.float64)
        swim_spd = np.zeros(nsegments, np.float64)
        for ns in range(nsegments):
            swim_spd[ns] = np.sqrt(swim_u[ns]*swim_u[ns]+swim_v[ns]*swim_v[ns])
            hydro_head[ns] = math.atan2(vh[ns],uh[ns])
            swim_head[ns] = math.atan2(swim_v[ns],swim_u[ns])
        self.swim_dframe = self.swim_dframe.assign(swim_u = swim_u)
        self.swim_dframe = self.swim_dframe.assign(swim_v = swim_v)
        self.swim_dframe = self.swim_dframe.assign(swim_spd = swim_spd)
        self.swim_dframe = self.swim_dframe.assign(hydro_head = hydro_head)
        self.swim_dframe = self.swim_dframe.assign(swim_head = swim_head)
        self.rec_swim = self.swim_dframe.to_records() 
        u_rel,v_rel = self.vel_from_hydro(vel=[swim_u,swim_v], 
                                          rec_swim=self.rec_swim)
        self.swim_dframe = self.swim_dframe.assign(swim_u_rel = u_rel)
        self.swim_dframe = self.swim_dframe.assign(swim_v_rel = v_rel)
        self.rec_swim = self.swim_dframe.to_records() 

        return

    # calculate autocorrelation for overall track
    def autocorr(self, rec_track=None, mask=None):
        """ computed difference in consecutive headings """
        if rec_track is None:
            rec_track = self.rec_track
        if mask is not None:
            valid = rec_track[mask]
            rec_seg = self.make_segments(input_rec_track = valid)
        else:
            valid = rec_track
            rec_seg = self.make_segments(input_rec_track = rec_track)
        del_vel = self.del_velocity(input_rec_track = valid)
        delt = np.diff(valid.Sec)
        plt.plot(delt, del_vel, '*')
        plt.ion()
        plt.show()
        pdb.set_trace()

        return

    # calculate turn angle between consecutive segments
    def turn_angle(self, rec_track=None, mask=None):
        """ computed difference in consecutive headings """
        if rec_track is None:
            rec_track = self.rec_track
        if mask is not None:
            valid = rec_track[mask]
            rec_seg = self.make_segments(input_rec_track = valid)
        else:
            rec_seg = self.make_segments(input_rec_track = rec_track)
        turn_angle = self.turn_angle_of_seg(rec_seg)
        if mask is None:
            self.df_track = self.df_track.assign(turn_angle = turn_angle)
            self.rec_track = self.df_track.to_records()
            return
        else:
            return turn_angle

    def turn_angle_of_seg(self, rec_seg=None):
        """ computer difference in consecutive headings """
        if rec_seg is None:
            rec_seg = self.rec_seg
        nseg = len(rec_seg)
        ndetects = nseg+1
        turn_angle = np.zeros(ndetects, np.float64)
        for nd in range(1,ndetects-2):
            seg1 = rec_seg[nd-1]
            seg2 = rec_seg[nd]
            dy = np.sin(seg1.head - seg2.head)
            dx = np.cos(seg1.head - seg2.head)
            turn_angle[nd] = math.atan2(dy, dx)

        return turn_angle

    # calculate circular std dev used by Potts
    def circular_std_dev(self, window=10, headings=None):
        if headings is None:
            seg = self.rec_seg
            headings = seg.head
        ndetects = len(headings) + 1
        halfres = int(window/2)
        circ_std = np.zeros(ndetects, np.float64)
        circ_var = np.zeros(ndetects, np.float64)
        circ_avg = np.zeros(ndetects, np.float64)
        for nd in range(1,ndetects):
            nd_min = max(0,nd-halfres)
            nd_max = min(ndetects-1,nd+halfres)
            if nd_max > nd_min+1:
                circ_std[nd] = circstd(headings[nd_min:nd_max]) # scipy
                circ_var[nd] = circ_std[nd]**2
                circ_avg[nd] = circmean(headings[nd_min:nd_max]) # scipy
        
        if headings is None:
            self.df_track = self.df_track.assign(circ_std = circ_std)
            self.df_track = self.df_track.assign(circ_var = circ_var)
            self.df_track = self.df_track.assign(circ_avg = circ_avg)
            self.rec_track = self.df_track.to_records()
        else:
            return circ_var, circ_avg

        return

    def identify_outliers(self, methods, params):
        """ identify outliers for a single track """ 
        self.outlier_methods = methods
        self.outlier_params = params
        nmethods = len(methods)
        flagged = np.zeros((nmethods, self.ndetects),np.int32)
        cross = np.zeros((self.ndetects), np.float64)
        if 'Potts' in methods:
            self.turn_angle() 
            self.circular_std_dev()
            circ_std = self.rec_track.circ_std
            turn_angle = abs(self.rec_track.turn_angle)
            turning = 0 # initialize
        if 'Dry' in methods: # need to do this first
            ndry = self.outlier_methods.index('Dry')
            for nd in range(self.ndetects):
                tr = self.rec_track[nd]
                if tr.i < 0:
                    flagged[ndry, nd] = True
                if tr.depth < params[ndry][0]:
                    flagged[ndry, nd] = True
            ncol = len(self.df_track.columns)
            self.df_track.insert(ncol, 'Dry', flagged[ndry,:])
            self.rec_track = self.df_track.to_records()
        if 'Iterative' in methods: # need to do this first
            nit = self.outlier_methods.index('Iterative')
            max_del_vel = self.outlier_params[nit][0]
            print("calling id outliers it")
            self.identify_outliers_iterative(max_del_vel)
        if 'Poly' in methods: # need to do this second
            # assumes that Dry and Iterative flags have been set
            nm = self.outlier_methods.index('Poly')
            if 'Dry' in methods and 'Iterative' in methods:
                tot_flagged = np.maximum(flagged[nit,:],flagged[ndry,:])
                not_flagged = np.where(tot_flagged == 0)[0]
            elif 'Dry' in methods:
                not_flagged = np.where(flagged[ndry,:] == 0)[0]
            elif 'Iterative' in methods:
                not_flagged = np.where(flagged[nit,:] == 0)[0]
            nm = self.outlier_methods.index('Poly')
            names = params[nm][0]
            polys = params[nm][1]
            ipoly = names.index('upstream')
            poly = polys[ipoly]
            # find last point in upstream poly
            last_upstream = 0

            #for nd in range(self.ndetects):
            for nd in not_flagged: # only look at potentially valid pts
                tr = self.rec_track[nd]
                pt = geometry.Point([tr.X,tr.Y])
                if pt.intersects(poly):
                    if not flagged[ndry,nd]:
                        last_upstream = nd
            flagged[nm,0:last_upstream+1] = 1
            # find first point in downstream poly
            ipoly = names.index('downstream')
            poly = polys[ipoly]
            # find first point in downstream poly
            first_downstream = self.ndetects-1
            for nd in range(self.ndetects):
                tr = self.rec_track[nd]
                pt = geometry.Point([tr.X,tr.Y])
                if pt.intersects(poly):
                    if not flagged[ndry,nd]:
                        first_downstream = nd
                        break
            flagged[nm,max(0,first_downstream-1):] = 1

        for nd in range(1,self.ndetects-1):
            seg1 = self.rec_seg[nd-1]
            seg2 = self.rec_seg[nd]
            tr = self.rec_track[nd]
            for nm, method in enumerate(methods):
                if method == 'Tek':
                    if seg1.speed > params[nm][0]:
                        flagged[nm, nd] = 1
                if method == 'Cross':
                    #cross = abs(seg1.u*seg2.v - seg1.v*seg2.u)
                    # change "cross" to vector difference
                    du = seg1.u - seg2.u
                    dv = seg1.v - seg2.v
                    cross[nd] = np.sqrt(du*du + dv*dv)
                    if cross[nd] > params[nm][0]:
                        if nd > 1:
                            if cross[nd] > cross_last:
                                flagged[nm, nd] = 1
                                flagged[nm, nd-1] = 0 
                    cross_last = cross[nd]
                if method == 'Potts':
                    ave_circ_std = np.average(circ_std) 
                    threshold = params[nm][0]*ave_circ_std
                    if circ_std[nd] > threshold:
                        nd_start = nd
                        turning = 1
                    if (circ_std[nd] < threshold) and (turning == 1):
                        turning = 0
                        #nd_change = int(np.rint((nd_start+nd)/2.0))
                        nd_change = nd_start+np.argmax(turn_angle[nd_start:nd])
                        flagged[nm, nd_change] = 1
        for nm, method in enumerate(methods):
            if method == 'Cross':
                ncol = len(self.df_track.columns)
                self.df_track.insert(ncol, 'del_vel', cross)
            if method not in ['Iterative','Dry']: 
                ncol = len(self.df_track.columns)
                self.df_track.insert(ncol, method, flagged[nm,:])

        self.rec_track = self.df_track.to_records()

        return

    def identify_route_selection_new(self, mask):
        #diffY = np.max(self.rec_track.Y[mask]) - 4185880
        #diffX = 647220 - np.min(self.rec_track.X[mask])
        diffY = self.rec_track.Y[mask[-1]] - 4185870
        diffX = 647230 - self.rec_track.X[mask[-1]]
        if (diffY <= 0) and (diffX <= 0.0):
            self.route = 'None'
        elif (diffY*diffX < 0.0):
            if (diffY > 0.0):
                self.route = 'SJ'
            else:
                self.route = 'Old'
        else:
            if (diffY > diffX):
                self.route = 'SJ'
            else:
                self.route = 'Old'

        return

    def identify_outliers_iterativeJUNK(self, max_del_vel):
        """ iteratively apply delta velocity method """
        pdb.set_trace()
        flagged = np.zeros(self.ndetects,np.int32)
        dv_final = np.zeros(self.ndetects,np.float64)
        if 'Dry' in self.outlier_methods:
            flagged = self.rec_track['Dry']
        # moved Iterative BEFORE Poly
#       if 'Poly' in self.outlier_methods:
#           flagged = np.logical_or(self.rec_track['Poly'], flagged)
        niterations = 5
        for nit in range(0,niterations):
            not_flagged = np.where(flagged==0)[0]
            valid_tr = self.rec_track[not_flagged]
            # compute segments associated with valid detects
            nvalid = len(valid_tr)
            if nvalid < 3:
                self.valid = False
            else:
                valid_seg = self.make_segments(input_rec_track = valid_tr)
                del_vel = self.del_velocity(input_rec_track = valid_tr)
                for nd in range(1,nvalid-1):
                    nd_orig = valid_tr[nd].nd
                    dv_final[nd_orig] = del_vel[nd]
                    if del_vel[nd] > max_del_vel:
                        nd_minus1_orig = valid_tr[nd-1].nd
                        flagged[nd_orig] = 1
                        if del_vel[nd] > del_vel[nd-1]:
                            flagged[nd_minus1_orig] = 0 

            pdb.set_trace()
        ncol = len(self.df_track.columns)

        return

    def identify_outliers_iterative(self, max_del_vel):
        """ iteratively apply delta velocity method """
        flagged = np.zeros(self.ndetects,np.int32)
        dv_final = np.zeros(self.ndetects,np.float64)
        if 'Dry' in self.outlier_methods:
            flagged = self.rec_track['Dry']
        # moved Iterative BEFORE Poly
        #       if 'Poly' in self.outlier_methods:
        #           flagged = np.logical_or(self.rec_track['Poly'], flagged)
        # del-velocity is one shorter, so leave last entry as 0
        # dv_final[i] is then change from i to i+1
        dv_final[:-1] = self.del_velocity() # initialize
        niterations = 5
        for nit in range(0,niterations):
            print("nit", nit)
            not_flagged = np.where(flagged==0)[0]
            valid_tr = self.rec_track[not_flagged]
            # compute segments associated with valid detects
            nvalid = len(valid_tr)
            if nvalid < 3:
                self.valid = False
            else:
                valid_seg = self.make_segments(input_rec_track = valid_tr)
                del_vel = self.del_velocity(input_rec_track = valid_tr)
                for nd in range(1,nvalid-1):
                    nd_orig = valid_tr[nd].nd
                    dv_final[nd_orig] = del_vel[nd]
                    if del_vel[nd] > max_del_vel:
                        nd_minus1_orig = valid_tr[nd-1].nd
                        flagged[nd_orig] = 1
                        if del_vel[nd] > del_vel[nd-1]:
                            flagged[nd_minus1_orig] = 0 
        ncol = len(self.df_track.columns)
        self.df_track.insert(ncol, 'Iterative', flagged)
        ncol = len(self.df_track.columns)
        self.df_track.insert(ncol, 'dv_final', dv_final)
        self.rec_track = self.df_track.to_records()

        return

    def identify_route_selection(self, mask):
        #diffY = np.max(self.rec_track.Y[mask]) - 4185880
        #diffX = 647220 - np.min(self.rec_track.X[mask])
        diffY = self.rec_track.Y[mask[-1]] - 4185870
        diffX = 647230 - self.rec_track.X[mask[-1]]
        if (diffY <= 0) and (diffX <= 0.0):
            self.route = 'None'
        elif (diffY*diffX < 0.0):
            if (diffY > 0.0):
                self.route = 'SJ'
            else:
                self.route = 'Old'
        else:
            if (diffY > diffX):
                self.route = 'SJ'
            else:
                self.route = 'Old'

        return

    def total_outlier_flag(self):
        """ set a flag indicating whether a detection is an outlier
            according to one or more of the chosen screening approaches """
        flagged = np.zeros(self.ndetects,np.int32)
        for nm, method in enumerate(self.outlier_methods):
            if hasattr(self.rec_track, method):
                flagged = np.maximum(flagged, self.rec_track[method])
        if 'flagged' in self.df_track.columns:
            self.df_track = self.df_track.drop(columns=['flagged'])
        ncol = len(self.df_track.columns)
        self.df_track.insert(ncol, 'flagged', flagged)
        self.rec_track = self.df_track.to_records()

        return

    def make_segments(self, set_depth=True, input_rec_track=None,
                      overwrite_rec_seg=False):
        """ set velocities associated with a single track 
            if input_rec_track is passed a segment record
            will be returned. Otherwise all detections will
            be analyzed and the segment record added to self """ 
        grd = self.grd
        if input_rec_track is None:
            rec_track = self.rec_track
        else:
            rec_track = input_rec_track
        ndetects = len(rec_track)
        nsegments = ndetects - 1
        self.nsegments = nsegments
        if nsegments < 1:
            return
        x1 = np.nan*np.ones(nsegments, np.float64)
        y1 = np.nan*np.ones(nsegments, np.float64)
        z1 = np.nan*np.ones(nsegments, np.float64)
        x2 = np.nan*np.ones(nsegments, np.float64)
        y2 = np.nan*np.ones(nsegments, np.float64)
        z2 = np.nan*np.ones(nsegments, np.float64)
        xm = np.nan*np.ones(nsegments, np.float64)
        ym = np.nan*np.ones(nsegments, np.float64)
        zm = np.nan*np.ones(nsegments, np.float64)
        t1 = np.nan*np.ones(nsegments, np.float64)
        t2 = np.nan*np.ones(nsegments, np.float64)
        tm = np.nan*np.ones(nsegments, np.float64)
        dn1 = np.nan*np.ones(nsegments, np.float64)
        dn2 = np.nan*np.ones(nsegments, np.float64)
        dnm = np.nan*np.ones(nsegments, np.float64)
        dt = np.nan*np.ones(nsegments, np.float64)
        dist = np.nan*np.ones(nsegments, np.float64)
        speed = np.nan*np.ones(nsegments, np.float64)
        u = np.nan*np.ones(nsegments, np.float64)
        v = np.nan*np.ones(nsegments, np.float64)
        head = np.nan*np.ones(nsegments, np.float64)
        idetect = np.nan*np.ones(nsegments, np.int32)
        tag_id = []
        for nd in range(1,ndetects):
            tag_id.append(self.ID)
            ns = nd-1
            x1[ns] = rec_track.X[nd-1]
            y1[ns] = rec_track.Y[nd-1]
            z1[ns] = rec_track.Z[nd-1]
            x2[ns] = rec_track.X[nd]
            y2[ns] = rec_track.Y[nd]
            z2[ns] = rec_track.Z[nd]
            t1[ns] = rec_track.Sec[nd-1]
            t2[ns] = rec_track.Sec[nd]
            dn1[ns] = rec_track.dnums[nd-1]
            dn2[ns] = rec_track.dnums[nd]
            dx = x2[ns] - x1[ns]
            dy = y2[ns] - y1[ns]
            xm[ns] = 0.5*(x1[ns]+x2[ns])
            ym[ns] = 0.5*(y1[ns]+y2[ns])
            zm[ns] = 0.5*(z1[ns]+z2[ns])
            tm[ns] = 0.5*(t1[ns]+t2[ns])
            dnm[ns] = 0.5*(dn1[ns]+dn2[ns])
            dt[ns] = t2[ns] - t1[ns]
            dist[ns] = np.sqrt(dx*dx + dy*dy)
            if dt[ns] > 0.0:
                speed[ns] = dist[ns]/dt[ns] 
                u[ns] = dx/dt[ns]
                v[ns] = dy/dt[ns]
            head[ns] = math.atan2(v[ns],u[ns])
        df_seg = pd.DataFrame({'id':np.asarray(tag_id), 
                               'x1':x1, 'x2':x2, 'xm':xm,
                               'y1':y1, 'y2':y2, 'ym':ym,
                               'z1':z1, 'z2':z2, 'zm':zm,
                               't1':t1, 't2':t2, 'tm':tm,
                               'dn1':dn1, 'dn2':dn2, 'dnm':dnm,
                               'dt':dt, 'dist':dist, 'head':head,
                               'speed':speed, 'u':u, 'v':v})
        if set_depth:
            depth = np.nan*np.ones(nsegments, np.float64)
            for ns in range(nsegments):
                xy = [xm[ns], ym[ns]]
                i = grd.select_cells_nearest(xy, inside=True)
                if i is None: i=-1
                idetect[ns] = i
                if i >= 0:
                    depth[ns] = grd.cells['depth'][i]
            df_seg = df_seg.assign(depth = depth)
            df_seg = df_seg.assign(i = idetect)

        rec_seg = df_seg.to_records()
        if (input_rec_track is None) or overwrite_rec_seg:
            self.df_seg = copy.deepcopy(df_seg)
            self.rec_seg = copy.deepcopy(rec_seg)
            return
        else:
            return rec_seg

#   def smoothed_vel(self, mask=None):
#       """ smoothed estimate of velocity by linear fitting 
#           estimate at ALL points even if only valid data 
#           are used in linear fits. """
#       dt_smooth = 30.
#       if mask is not None:
#           rec_track = self.rec_track[mask]
#           rec_seg = self.make_segments(set_depth=True, 
#                                        input_rec_track=rec_track)
#           nsegments = len(rec_seg)
#       else:
#           rec_track = self.rec_track
#           nsegments = self.nsegments
#           rec_seg = self.rec_seg
#       us = np.nan*np.ones(nsegments, np.float64)
#       vs = np.nan*np.ones(nsegments, np.float64)
#       ss = np.nan*np.ones(nsegments, np.float64)
#       ud = np.nan*np.ones(nsegments, np.float64)
#       vd = np.nan*np.ones(nsegments, np.float64)
#       sd = np.nan*np.ones(nsegments, np.float64)
#       nsmooth = np.nan*np.ones(nsegments, np.float64)
#       tm = rec_seg.tm
#       td = rec_track.Sec
#       u = rec_seg.u
#       v = rec_seg.v
#       speed = rec_seg.speed
#       for ns in range(nsegments):
#           t0 = tm[ns] - dt_smooth
#           t1 = tm[ns] + dt_smooth
#           idet = np.where((td >= t0) & (td <= t1))[0]
#           nd = len(idet)
#           nsmooth[ns] = nd
#           if nd > 4:
#               xs = rec_track.X[idet]
#               ys = rec_track.Y[idet]
#               ts = rec_track.Sec[idet]
#               m, b, r, p, se = stats.linregress(ts, xs)
#               us[ns] = m
#               ud[ns] = us[ns] - u[ns]
#               m, b, r, p, se = stats.linregress(ts, ys)
#               vs[ns] = m
#               vd[ns] = vs[ns] - v[ns]
#               ss[ns] = np.sqrt(us[ns]*us[ns] + vs[ns]*vs[ns])
#               sd[ns] = ss[ns] - speed[ns]
#       self.df_smooth = pd.DataFrame({'us':us, 'vs':vs, 'ss':ss,
#                                      'ud':ud, 'vd':vd, 's':sd,
#                                      'nsmooth':nsmooth})
#       self.rec_smooth = self.df_smooth.to_records()

#       return

    def find_change_points(self):
        # assume rec_smooth_pos is already set
        turn_threshold = np.pi/3.  # hardwired threshold
        del_vel_threshold = 0.75  # hardwired threshold
        rec_seg = self.make_segments(set_depth=True, 
                                     input_rec_track=self.rec_smooth_pos)
        turn_angle = self.turn_angle_of_seg(rec_seg)
        self.df_smooth_pos = self.df_smooth_pos.assign(turn_angle = turn_angle)
        ndetects = len(self.rec_smooth_pos)
        change_pt_flag2 = np.zeros(ndetects, np.bool_)
        change_pt_flag3 = np.zeros(ndetects, np.bool_)
        change_pt_flag1 = turn_angle > turn_threshold
        self.df_smooth_pos = self.df_smooth_pos.assign(change_pt_flag1=change_pt_flag1)
        turn_dt = np.add(rec_seg.dt[0:-1],rec_seg.dt[1:])
        turn_per_dt = np.divide(turn_angle[1:-1], turn_dt)
        change_pt_flag2[1:-1] = turn_per_dt > turn_threshold/5. # 5 second interval
        self.df_smooth_pos = self.df_smooth_pos.assign(change_pt_flag2=change_pt_flag2)
        del_vel = self.del_velocity(input_rec_track = self.rec_smooth_pos)
        change_pt_flag3[1:] = del_vel > del_vel_threshold 
        self.df_smooth_pos = self.df_smooth_pos.assign(change_pt_flag3=change_pt_flag3)
        self.rec_smooth_pos = self.df_smooth_pos.to_records()
        
        return

    def find_change_points_fill(self):
        """ find change points in smoothed and filled position set """
        # Fill data with moving window of mean heading and circular stdev
        # - get filled data at 5 second interval
        # - calculate headings of filled data
        # - calculate mean and circular stdev in moving window. 
        # - look for points where forward window mean differs from backward
        #   window mean by more than 2 stdev
        rec_fill = self.rec_smooth_fill
        segs = self.make_segments(set_depth=True, input_rec_track=rec_fill,
                                  overwrite_rec_seg=False)
        # need rec_track to get valid data points for t-statistic calculation
        rec_raw = self.rec_track
        not_flagged = np.where(rec_raw.flagged==0)[0]
        rec_raw_valid = rec_raw[not_flagged]
        nfill = len(rec_fill)
        nseg = nfill - 1
        window = 12
        cvar, cavg = self.circular_std_dev(window=int(window/2), headings=segs.head)
        change_pt = np.zeros(nfill, np.bool_)
        change_pt0 = np.zeros(nfill, np.bool_)
        p_stat = np.ones(nfill, np.float64)
        small_cstd = 0.1*np.pi
        for nf in range(int(window/2+1),int(nfill-window/2-1)):
            nf_before = int(nf - window/4)
            nf_after  = int(nf + window/4)
            cb = cavg[nf_before]
            ca = cavg[nf_after]
            #dy = np.sin(cb - ca)
            #dx = np.cos(cb - ca)
            #del_avg = math.atan2(dy, dx)
            del_avg = (cb-ca+np.pi)%(2.0*np.pi)-np.pi
            #avg_var = np.average([cvar[nf_before], cvar[nf_after]])
            #cstd = max(np.sqrt(avg_var),small_cstd)
            t1 = segs.t1[nf_before]
            t2 = segs.t2[nf_after]
            tf = rec_fill.Sec[nf]
            td = rec_raw_valid.Sec
            # calculate sample size
            n1 = max(1, np.sum(np.logical_and(td >= t1, td <= tf)))
            n2 = max(1, np.sum(np.logical_and(td <= t2, td >= tf)))
            #idet = np.where((td >= t1) & (td <= t2))[0]
            #nd = len(idet) # number of original detections in window
            # http://www.stat.yale.edu/Courses/1997-98/101/meancomp.htm
            # degrees of freedom is min(n1,n2)-1
            #df = 2*nd - 2
            #df = max(1, df)
            #t_stat = del_avg/(cstd*np.sqrt(df))
            #avg_var = np.average([cvar[nf_before], cvar[nf_after]])
            denominator = np.sqrt(cvar[nf_before]/n1 + cvar[nf_after]/n2)
            denominator = max(0.05*np.pi, denominator)
            t_stat = abs(del_avg)/denominator
            df = max(1, min(n1, n2)-1)
            p_stat[nf] = 1 - stats.t.cdf(t_stat, df=df)
            #print self.ID, nd, df, del_avg, cstd, t_stat, p_stat[nf]
            #change_pt[nf] = abs(del_avg) > 3.*avg_std
            change_pt0[nf] = p_stat[nf] < 0.05
        for nf in range(int(window/2+1),int(nfill-window/2-1)):
            # flag current point only if it is the smallest p_stat in window
            if change_pt0[nf]:
                nf_before = nf - int(window/4)
                nf_after  = nf + int(window/4)
                nf_min = np.argmin(p_stat[nf_before:nf_after])
                if nf == nf_min + nf_before:
                    change_pt[nf] = True

        self.df_smooth_fill = self.df_smooth_fill.assign(change_pt=change_pt)
        self.df_smooth_fill = self.df_smooth_fill.assign(p_stat=p_stat)
        self.rec_smooth_fill = self.df_smooth_fill.to_records()

        return

    def smoothed_vel_npts(self, mask=None, npts=4):
        """ smoothed estimate of velocity by linear fitting 
            use fixed number of points in fitting even if not close 
            in time """
        if mask is not None:
            rec_track = self.rec_track[mask]
            if len(rec_track) < 3:
                self.valid = False
                return
            rec_seg = self.make_segments(set_depth=True, 
                                         input_rec_track=rec_track)
            nsegments = len(rec_seg)
        else:
            rec_track = self.rec_track
            nsegments = self.nsegments
            rec_seg = self.rec_seg
        us = np.nan*np.ones(nsegments, np.float64)
        vs = np.nan*np.ones(nsegments, np.float64)
        ss = np.nan*np.ones(nsegments, np.float64)
        ud = np.nan*np.ones(nsegments, np.float64)
        vd = np.nan*np.ones(nsegments, np.float64)
        sd = np.nan*np.ones(nsegments, np.float64)
        nsmooth = np.nan*np.ones(nsegments, np.float64)
        tm = rec_seg.tm
        td = rec_track.Sec
        u = rec_seg.u
        v = rec_seg.v
        speed = rec_seg.speed
        for ns in range(nsegments):
            ibefore = np.where(td <= tm[ns])[0]
            iafter = np.where(td > tm[ns])[0] 
            nbefore = len(ibefore)
            nafter = len(iafter)
            idet = ibefore[-min(nbefore,npts):]
            idet = np.concatenate((idet, iafter[0:min(nafter,npts):]))
            nd = len(idet)
            nsmooth[ns] = nd
            xs = rec_track.X[idet]
            ys = rec_track.Y[idet]
            ts = rec_track.Sec[idet]
            m, b, r, p, se = stats.linregress(ts, xs)
            us[ns] = m
            ud[ns] = us[ns] - u[ns]
            m, b, r, p, se = stats.linregress(ts, ys)
            vs[ns] = m
            vd[ns] = vs[ns] - v[ns]
            ss[ns] = np.sqrt(us[ns]*us[ns] + vs[ns]*vs[ns])
            sd[ns] = ss[ns] - speed[ns]
        self.df_smooth = pd.DataFrame({'us':us, 'vs':vs, 'ss':ss,
                                       'ud':ud, 'vd':vd, 'sd':sd,
                                       'tm':tm,
                                       'nsmooth':nsmooth})

        self.rec_smooth = self.df_smooth.to_records()
        return 

#   def smoothed_pos(self, mask=None, input_rec_track=None,
#                    window='centered'):
#       """ smoothed estimate of position by linear fitting """
#       dt_smooth = 60.
#       xs = np.nan*np.ones(self.ndetects, np.float64)
#       ys = np.nan*np.ones(self.ndetects, np.float64)
#       xd = np.nan*np.ones(self.ndetects, np.float64)
#       yd = np.nan*np.ones(self.ndetects, np.float64)
#       noise = np.nan*np.ones(self.ndetects, np.float64)
#       nsmooth = np.nan*np.ones(self.ndetects, np.float64)
#       if input_rec_track == None:
#           rec_track = self.rec_track
#       else:
#           rec_track = input_rec_track
#       td = rec_track.Sec
#       x = rec_track.X
#       y = rec_track.Y
#       if mask is None:
#           ndetects = len(x)
#           tdm = rec_track.Sec
#           xm = rec_track.X
#           ym = rec_track.Y
#       else:
#           tdm = rec_track.Sec[mask]
#           xm = rec_track.X[mask]
#           ym = rec_track.Y[mask]
#       print self.ID, self.ndetects, len(td)
#       # currently smoothes at all locations in ORIGINAL data
#       # should have option to estimate locations at uniform interval
#       # perhaps save rec_good in ADDITION to rec_track
#       # then pass one or the other (rec_track or rec_good) in to this method
#       for nd in range(self.ndetects):
#           t0 = td[nd] - dt_smooth
#           t1 = td[nd] + dt_smooth
#           # idet holds indices of MASKED array
#           idet = np.where((tdm >= t0) & (tdm <= t1))[0]
#           ndet = len(idet)
#           nsmooth[nd] = ndet
#           if ndet > 4:
#               tsm = tdm[idet]
#               m, b, r, p, se = stats.linregress(tsm, xm[idet])
#               xs[nd] = m*td[nd] + b 
#               xd[nd] = xs[nd] - x[nd]
#               m, b, r, p, se = stats.linregress(tsm, ym[idet])
#               ys[nd] = m*td[nd] + b 
#               yd[nd] = ys[nd] - y[nd]
#               noise[nd] = np.sqrt(xd[nd]*xd[nd] + yd[nd]*yd[nd])
#       self.df_smooth_pos = pd.DataFrame({'xs':xs, 'ys':ys,
#                                          'xd':xd, 'yd':yd, 
#                                          'nsmooth':nsmooth,
#                                          'noise':noise})
#       rec_smooth_pos = self.df_smooth_pos.to_records()
#       if input_rec_track is None:
#           self.rec_smooth_pos = rec_smooth_pos
#           return
#       else:
#           return rec_smooth_pos

    def smoothed_pos_npts(self, mask=None, input_rec_track=None,
                     window='centered', npts=3):
        """ smoothed estimate of position by linear fitting """
        if not self.valid:
            return
        if input_rec_track == None:
            rec_track = self.rec_track
        else:
            rec_track = input_rec_track

        if mask is None:
            ndetects = len(rec_track)
            mask = np.arange(1,ndetects)
        tdm = rec_track.Sec[mask]
        xm = rec_track.X[mask]
        ym = rec_track.Y[mask]
        zm = rec_track.Z[mask]
        
        ndetects = len(tdm)
        xs = np.nan*np.ones(ndetects, np.float64)
        ys = np.nan*np.ones(ndetects, np.float64)
        zs = np.nan*np.ones(ndetects, np.float64)
        xd = np.nan*np.ones(ndetects, np.float64)
        yd = np.nan*np.ones(ndetects, np.float64)
        zd = np.nan*np.ones(ndetects, np.float64)
        noise = np.nan*np.ones(ndetects, np.float64)
        nsmooth = np.nan*np.ones(ndetects, np.int32)
        for nd in range(ndetects):
            idet = np.arange(max(0,nd-npts),min(ndetects-1,nd+npts))
            ndet = len(idet)
            nsmooth[nd] = ndet
            tsm = tdm[idet]
            if ndet < 1: # give up
                xs[nd] = xm[nd]
                ys[nd] = ym[nd]
                zs[nd] = zm[nd]
            else:
                m, b, r, p, se = stats.linregress(tsm, xm[idet])
                xs[nd] = m*tdm[nd] + b 
                m, b, r, p, se = stats.linregress(tsm, ym[idet])
                ys[nd] = m*tdm[nd] + b 
                m, b, r, p, se = stats.linregress(tsm, zm[idet])
                zs[nd] = m*tdm[nd] + b 
            xd[nd] = xs[nd] - xm[nd]
            yd[nd] = ys[nd] - ym[nd]
            zd[nd] = zs[nd] - zm[nd]
            # Exclude Z from noise estimate -- no sense yet of how
            # good/bad Z is.
            noise[nd] = np.sqrt(xd[nd]*xd[nd] + yd[nd]*yd[nd])
        # variables for filled and smoothed track
        duration = tdm[-1] - tdm[0]
        nfilled = int(np.rint(duration/5.)) + 1
        xf = np.nan*np.ones(nfilled, np.float64)
        yf = np.nan*np.ones(nfilled, np.float64)
        zf = np.nan*np.ones(nfilled, np.float64)
        tf = np.nan*np.ones(nfilled, np.float64)
        dnumf = np.nan*np.ones(nfilled, np.float64)
        tint = 5.
        for nf in range(nfilled):
            tf[nf] = tdm[0] + nf*tint # 5 second interval
            dnumf[nf] = rec_track.dnums[0] + tint/86400.
        nd = 0
        nd_last = -1
        for nf in range(nfilled):
            if tf[nf] > tdm[nd]: # only need to increment once (no while)
                if nd < ndetects-2:
                    nd += 1          
            if nd != nd_last:
                idet = np.arange(max(0,nd-npts),min(ndetects-1,nd+npts))
                ndet = len(idet)
                tsm = tdm[idet]
                if ndet >= 1:
                    mx, bx, r, p, se = stats.linregress(tsm, xm[idet])
                    my, by, r, p, se = stats.linregress(tsm, ym[idet])
                    mz, bz, r, p, se = stats.linregress(tsm, zm[idet])
                nd_last = nd
            if ndet < 1: # give up
                xf[nf] = xm[nd]
                yf[nf] = ym[nd]
                zf[nf] = zm[nd]
            else:
                xf[nf] = mx*tf[nf] + bx
                yf[nf] = my*tf[nf] + by 
                zf[nf] = mz*tf[nf] + bz 

        self.df_smooth_pos = pd.DataFrame({'X':xs, 'Y':ys, 'Z':zs,
                                           'Sec':rec_track.Sec[mask], 
                                           'dnums':rec_track.dnums[mask],
                                           'xd':xd, 'yd':yd, 'zd':zd,
                                           'nsmooth':nsmooth,
                                           'noise':noise})
        rec_smooth_pos = self.df_smooth_pos.to_records()
        self.df_smooth_fill = pd.DataFrame({'X':xf, 'Y':yf, 'Z':zf,
                                            'Sec':tf, 
                                            'dnums':dnumf})
        # set self.rec_smooth_fill always (even if input_rec_track is given)
        self.rec_smooth_fill = self.df_smooth_fill.to_records()
        if input_rec_track is None:
            self.rec_smooth_pos = rec_smooth_pos
            return
        else:
            return rec_smooth_pos

    def nondetects(self, masked=False):
        """ estimate coordinates of nondetects """
        grd = self.grd
        xnd = []
        ynd = []
        ncells = len(grd.cells['depth'])
        non_detects_i_tr = np.zeros(ncells, np.int32)
        if masked:
            not_flagged = np.where(self.rec_track.flagged==0)[0]
            rec_track = self.rec_track[not_flagged]
            rec_seg = self.make_segments(set_depth=True, 
                                         input_rec_track=rec_track)
        else:
            rec_seg = self.rec_seg
        for nr, rseg in enumerate(rec_seg):
            seg = rec_seg[nr]
            dt = seg.dt
            if dt > dt_signal+1:
                t1 = seg.t1
                t2 = seg.t2
                nint = int(np.rint((t2-t1)/dt_signal)) - 1
                x1 = seg.x1
                x2 = seg.x2
                y1 = seg.y1
                y2 = seg.y2
                dx_nd = (x2 - x1)/float(nint+1)
                dy_nd = (y2 - y1)/float(nint+1)
                if nint < 120: # 10 minute cutoff for nondetect filling
                    xint = [x1 + n*dx_nd for n in range(1,nint)]
                    yint = [y1 + n*dy_nd for n in range(1,nint)]
                    xnd = xnd + xint
                    ynd = ynd + yint

        for nd in range(len(xnd)):
            xy = [xnd[nd], ynd[nd]]
            i = grd.select_cells_nearest(xy)
            if (i is not None) and (i >= 0):
                non_detects_i_tr[i] += 1

        return non_detects_i_tr

    def track_quality_metrics(self, masked=False):
        """ calculate and save metrics of overall track quality """
        grd = self.grd
        if masked:
            not_flagged = np.where(self.rec_track.flagged==0)[0]
            tr = self.rec_track[not_flagged]
            # smooth track
            #trs = self.rec_smooth_pos[not_flagged]
            # assume that smoothed position is already masked
            trs = self.rec_smooth_pos
            seg = self.make_segments(set_depth=True, input_rec_track=tr)
            mask = not_flagged
        else:
            tr = self.rec_track
            trs = self.rec_smooth_pos
            seg = self.rec_seg
            mask = None
        detects = len(tr.X)
        possible_detects = (tr.Sec[-1] - tr.Sec[0])/dt_signal
        non_detects = possible_detects - detects
        frac_detects = float(detects)/float(possible_detects)
        max_depth = np.max(tr.depth)
        z_over_H = np.minimum(1.,np.divide(tr.Z,np.maximum(depth_min,tr.depth)))
        ivalid = np.where(np.isfinite(z_over_H))[0]
        avg_ht = np.average(1.0 - z_over_H[ivalid])
        avg_z = np.average(tr.Z)
        valid_depth = np.where(np.isfinite(tr.depth))[0]
        avg_depth = np.average(tr.depth[valid_depth])
        velocity_sd = np.std(seg.speed)
        dv = self.del_velocity(input_rec_track=trs)
        del_velocity_mag = np.average(np.abs(dv))
        del_velocity_sd = np.std(np.abs(dv))
        max_speed = np.max(np.absolute(seg.speed))
        detects_outside_grid = self.detects_outside_grid()
        t_elapsed = tr.Sec[-1] - tr.Sec[0]
        ivalid = np.where(np.isfinite(trs.noise))[0]
        pos_error = np.average(trs.noise[ivalid])
        t_entry = tr.dnums[0]
        hour = (t_entry - math.trunc(t_entry))*24.
        if (hour > 7) and (hour < 19):
            daytime_entry = True
        else:
            daytime_entry = False
        mnames = ['ID', 'ndetects','non_detects', 'frac_detects', 'max_depth', 
                  'avg_depth', 'avg_ht', 'avg_z', 'vel_sd', 
                  'dvel_mag', 'dvel_sd', 'invalid_cell', 'pos_error', 
                  't_elapsed','daytime_entry']
        metrics = [self.ID, detects, non_detects, frac_detects, max_depth, 
                   avg_depth, avg_ht, avg_z, velocity_sd, del_velocity_mag, 
                   del_velocity_sd, detects_outside_grid, pos_error, 
                   t_elapsed, daytime_entry]
        self.mnames = mnames
        self.metrics = metrics

        return mnames, metrics
        #print self.ID, non_detects, frac_detects, min_depth, velocity_sd, \
        #      detects_outside_grid

    def detects_outside_grid(self):
        """ detections outside of grid cells of specified grid """
        ii = self.rec_track['i']
        outside = sum(np.isnan(ii))

        return outside

#   def calc_del_vel(self, valid_tr, valid_seg):
#       nvalid = len(valid_tr)
#       del_vel = np.zeros(nvalid, np.float64)
#       for nd in range(1,nvalid-1):
#           seg1 = valid_seg[nd-1]
#           seg2 = valid_seg[nd]
#           du = seg1.u - seg2.u
#           dv = seg1.v - seg2.v
#           del_vel[nd] = np.sqrt(du*du + dv*dv)

#       return del_vel

    def del_velocity(self, mask=None, input_rec_track=None):
        """ compute metrics of segment to segment velocity variability """
        if input_rec_track is None:
            self.nsegments = self.ndetects - 1
            rec_track = self.rec_track
        else:
            rec_track = input_rec_track
        if mask is not None:
            rec_track = rec_track[mask]
        rec_seg = self.make_segments(input_rec_track = rec_track)
        nseg = len(rec_seg)
        dspeed = np.zeros(nseg, np.float64)
        for ns in range(nseg-1):
            seg1 = rec_seg[ns]
            seg2 = rec_seg[ns+1]
            du  = seg1.u - seg2.u
            dv  = seg1.v - seg2.v
            dspeed[ns] = np.sqrt(du*du + dv*dv)

        return dspeed

    def dir_from_hydro(self, mask=None, rec_swim=None):
        """ direction relative to hydrodynamic velocity """
        if rec_swim == None:
            rec_swim = self.rec_swim
        if mask is not None:
            rec_swim = rec_swim[mask]
        nseg = len(rec_swim)
        swim_dir = np.zeros(nseg, np.float64)
        for ns in range(nseg):
            dy = np.sin(rec_swim[ns].hydro_head - rec_swim[ns].swim_head)
            dx = np.cos(rec_swim[ns].hydro_head - rec_swim[ns].swim_head)
            swim_dir[ns] = math.atan2(dy, dx)

        return swim_dir

    def vel_from_hydro(self, vel=None, rec_swim=None):
        """ computer metrics of segment to segment velocity variability """
        u = vel[0]
        v = vel[1]
        nseg = len(vel[0])
        rel_dir = np.zeros(nseg, np.float64)
        u_rel = np.zeros(nseg, np.float64)
        v_rel = np.zeros(nseg, np.float64)
        for ns in range(nseg):
            vel_head = math.atan2(v[ns],u[ns])
            dy = np.sin(rec_swim[ns].hydro_head - vel_head)
            dx = np.cos(rec_swim[ns].hydro_head - vel_head)
            rel_dir[ns] = math.atan2(dy, dx)
            spd = np.sqrt(u[ns]*u[ns] + v[ns]*v[ns])
            u_rel[ns] = spd*np.cos(rel_dir[ns])
            # sign such that positive v is to the left of positive u
            v_rel[ns] = -spd*np.sin(rel_dir[ns]) 
            
        return u_rel, v_rel

    def cell_detects(self, masked=False):
        """ count number of detects in each grid cell """
        grd = self.grd
        ncells = len(grd.cells['depth'])
        detects_i_tr = np.zeros(ncells, np.int32)
        if masked:
            not_flagged = np.where(self.rec_track.flagged==0)[0]
            rec_track = self.rec_track[not_flagged]
        else:
            rec_track = self.rec_track
        ndetects = len(rec_track)
        for nd in range(ndetects):
            tr = rec_track[nd]
            i = tr.i
            if i >= 0:
                detects_i_tr[i] += 1
           
        return detects_i_tr

    def cell_speed(self):
        """ bin speed estimates by grid cell """
        grd = self.grd
        ncells = len(grd.cells['depth'])
        detects_i_tr = np.zeros(ncells, np.int32)
        speed_sum_i_tr = np.zeros(ncells, np.float64)
        for ns in range(self.nsegments):
            seg = self.rec_seg[ns]
            if seg.i >= 0:
                i = int(seg.i)
                detects_i_tr[i] += 1
                speed_sum_i_tr[i] += seg.speed
           
        return detects_i_tr, speed_sum_i_tr

    def cell_occupancy(self, masked=False):
        """ for each cell calculate whether the fish/tag occupied the 
            cell during the track """
        grd = self.grd
        occupancy_i = []
        if masked:
            not_flagged = np.where(self.rec_track.flagged==0)[0]
            rec_track = self.rec_track[not_flagged]
        else:
            rec_track = self.rec_track
        ndetects = len(rec_track)
        for nd in range(ndetects):
            tr = rec_track[nd]
            i = tr.i
            if i >= 0:
                occupancy_i.append(i)

        occupancy_i = np.unique(occupancy_i)

        return occupancy_i

    def plot_speed(self, variable, ax, color_by=None, mask=None, **kwargs):
        """ plot vectors of velocity over ground """
        if mask is not None:
            valid = self.rec_track[mask]
            seg = self.make_segments(input_rec_track = valid)
        else:
            seg = self.rec_seg
        nseg = len(seg)
        hlen = 4.0
        hwid = 3.0
        qscale = 0.05
        if variable is 'speed_over_ground':
            u = seg.u
            v = seg.v
            xm = seg.xm
            ym = seg.ym
        elif variable is 'smoothed_vel':
            smooth = self.rec_smooth
            u = smooth.us
            v = smooth.vs
            xm = self.rec_seg.xm
            ym = self.rec_seg.ym
        elif variable is 'swimming_speed':
            u = self.rec_swim.swim_u
            v = self.rec_swim.swim_v
            xm = self.rec_swim.xm
            ym = self.rec_swim.ym
            qscale = 0.02
        elif variable is 'hydro_speed':
            u = self.rec_swim.model_u
            v = self.rec_swim.model_v
            xm = self.rec_swim.xm
            ym = self.rec_swim.ym
        if color_by is None:
            self.quiv = ax.quiver(xm, ym, u, v, 
                                  units='x', scale=qscale, color='r',
                                  scale_units='x',
                                  headlength=hlen, headwidth=hwid)
        else:
            if color_by == 'age':
                # offset time by 1 second
                time_from_entry = seg.tm - self.t_entry + 1
                log_age = np.log10(time_from_entry)
                color = log_age
                cmap_quiv = cm_age
                label = 'Time from Entry (seconds)'
                color = log_age
            #else if color_by = 'quality':
            self.quiv = ax.quiver(seg.xm, seg.ym, u, v, color,
                                  units='x', scale=qscale, cmap=cmap_quiv, 
                                  headlength=hlen, headwidth=hwid)
            fig = plt.gcf()
            c1 = fig.colorbar(self.quiv)
            clims = [ticks[0],ticks[-1]]
            self.quiv.set_clim(clims)
            c1.set_label(label)
            c1.set_ticks(ticks)
            c1.set_ticklabels(tick_labels)

    def plot_vel_ts(self, variable, ax, **kwargs):
        """ plot u, v time series for a track """
        seg = self.rec_seg
        nseg = self.nsegments
        if variable is 'speed_over_ground':
            u = seg.u
            v = seg.v

        elif variable is 'smoothed_vel':
            smooth = self.rec_smooth
            u = smooth.us
            v = smooth.vs
        time_from_entry = seg.tm - self.t_entry
        kwargs['linestyle']='-'
        kwargs['linewidth']=0.5
        kwargs['markersize']=0.8
        ax.plot(time_from_entry, seg.u, marker='o',color='c', 
                label='u',**kwargs)
        ax.plot(time_from_entry, seg.v, marker='s',color='b', 
                label='v',**kwargs)
        ax.set_xlabel('time (seconds)')
        ax.set_ylim([-2, 2])
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.set_ylabel('velocity (m s$^{-1}$)')
        ax.legend(loc='upper right')

        return

    def plot_del_vel_histograms(self, ax1, ax2, turn_angle, 
                                masked=False, **kwargs):
        """ plot histogram of magnitude of velocity change """
        if masked == True:
            not_flagged = np.where(self.rec_track.flagged==0)[0]
            tr = self.rec_track[not_flagged]
        else:
            tr = self.rec_track
        ndetects = len(tr)
        dv = self.del_velocity(input_rec_track=tr)
        dv_mag = np.abs(dv)

        ax1.hist(turn_angle, bins=50, normed=False, facecolor='g')
        ax2.hist(dv_mag, bins=50, normed=False, facecolor='g')
        ax1.set_xlabel('turn angle (radians)')
        ax2.set_xlabel('change in velocity (m s$^{-1}$)')
        ax2.set_xlim([0,4])
        ax1.text(0.65, 0.86, "n = %d"%ndetects, transform=ax1.transAxes)
        ax1.text(0.65, 0.72, "max = %4.2f"%np.max(np.abs(turn_angle)), transform=ax1.transAxes)
        ax2.text(0.65, 0.86, "n = %d"%ndetects, transform=ax2.transAxes)
        ax2.text(0.65, 0.72, "max = %4.2f"%np.max(dv_mag), transform=ax2.transAxes)

        return 

    def plot_swim_vel_histograms(self, ax1, ax2,
                                 masked=False, **kwargs):
        """ plot histogram of direction and speed relative to hydro """
        nsegments = len(self.rec_swim)
        swim_dir = self.dir_from_hydro(rec_swim=self.rec_swim)
        swim_spd = self.rec_swim['swim_spd']

        ax1.hist(swim_dir, bins=50, normed=False, facecolor='g')
        ax2.hist(swim_spd, bins=50, normed=False, facecolor='g')
        ax1.set_xlabel('swim direction (radians)')
        ax2.set_xlabel('swim speed (m s$^{-1}$)')
        ax2.set_xlim([0,1])
        ax1.text(0.65, 0.86, "n = %d"%nsegments, transform=ax1.transAxes)
#       ax1.text(0.65, 0.72, "max = %4.2f"%np.max(np.abs(swim_dir)), transform=ax1.transAxes)
        ax2.text(0.65, 0.86, "avg = %4.2f"%np.average(swim_spd),
                                        transform=ax2.transAxes)
        ax2.text(0.65, 0.72, "max = %4.2f"%np.max(swim_spd), 
                                           transform=ax2.transAxes)

        return 

    def plot_swim_vel_scatter(self, ax1, ax2, masked=False, 
                              add_cbar=True, **kwargs):
        """ plot scatter of swim velocity in coordinates relative to hydro """
        rec_swim = self.rec_swim
        nsegments = len(rec_swim)

        swim_u = rec_swim.swim_u
        swim_v = rec_swim.swim_v
        u_rel,v_rel = self.vel_from_hydro(vel=[swim_u,swim_v], 
                                          rec_swim=rec_swim)
        #swim_spd = self.rec_swim['swim_spd']

        if nsegments==0:
            print("Tag %s has no segments. Skipping plot_swim_vel_scatter"%self.ID)
            return
        time_from_entry = rec_swim.tm - rec_swim.tm[0] + 1
        log_age = np.log10(time_from_entry)
        scat = ax1.scatter(u_rel, v_rel, marker='o', s=2.0, c=log_age)
        ax1.set_xlabel('swim u (m s$^{-1}$)')
        ax1.set_ylabel('swim v (m s$^{-1}$)')
        fsize = 6
        ax1.text(1.01, 0.90, "n = %d"%nsegments, transform=ax1.transAxes,
                 fontsize=fsize)
        ax1.text(1.01, 0.80, "avg|u| = %4.2f"%np.average(np.abs(u_rel)),
                 transform=ax1.transAxes, fontsize=fsize)
        ax1.text(1.01, 0.70, "avg|v| = %4.2f"%np.average(np.abs(v_rel)),
                 transform=ax1.transAxes, fontsize=fsize)
        ax1.set_xlim([-0.5,0.5])
        ax1.set_ylim([-0.5,0.5])
        ax1.axhline(0, color='k',zorder=0)
        ax1.axvline(0, color='k',zorder=0)
        ax1.set_aspect('equal')
        clims = [ticks[0],ticks[-1]]
        scat.set_clim(clims)
        if add_cbar:
            cax = plt.gcf().add_axes([0.41,0.12,0.16,0.025]) # hardwired
            label = 'Time (seconds)'
            c1 = plt.gcf().colorbar(scat, cax=cax, orientation='horizontal')
            #c1 = plt.gcf().colorbar(scat)
            c1.set_label(label)
            c1.set_ticks(ticks)
            c1.set_ticklabels(tick_labels)
        # smoothed speed over ground
        us = rec_swim.us
        vs = rec_swim.vs
        uh,vh = self.vel_from_hydro(vel=[us,vs], rec_swim=rec_swim)
        scat2 = ax2.scatter(uh, vh, marker='o', s=2.0, c=log_age)
        ax2.set_xlabel('u (m s$^{-1}$)')
        ax2.set_ylabel('v (m s$^{-1}$)')
        ax2.text(1.01, 0.90, "n = %d"%nsegments, transform=ax2.transAxes,
                 fontsize=fsize)
        ax2.text(1.01, 0.80, "avg|u| = %4.2f"%np.average(np.abs(uh)),
                 transform=ax2.transAxes, fontsize=fsize)
        ax2.text(1.01, 0.70, "avg|v| = %4.2f"%np.average(np.abs(vh)),
                 transform=ax2.transAxes, fontsize=fsize)
        ax2.axhline(0, color='k',zorder=0)
        ax2.axvline(0, color='k',zorder=0)
        ax2.set_xlim([-1,1])
        ax2.set_ylim([-1,1])
        ax2.set_aspect('equal')
        scat2.set_clim(clims)

        return 

    def plot_noise_distribution(self, ax, mask=None, **kwargs):
        """ plot histogram of estimated position error """
        trs = self.rec_smooth_pos
        ivalid = np.where(np.isfinite(trs.noise))[0]
        sigma = np.std(trs.noise[ivalid])
        mu = np.average(trs.noise[ivalid])

        kwargs['linestyle']='-'
        kwargs['linewidth']=1.0
        # fit lognormal
        if len(ivalid) > 10:
            try:
                x = np.linspace(-4.*sigma+mu, 4*sigma+mu, 200)
                params = stats.lognorm.fit(trs.noise[ivalid], loc=0.2)
                pdf_fit = stats.lognorm.pdf(x, params[0], loc=params[1], 
                                            scale=params[2])
                ax.plot(x, pdf_fit, color='k', **kwargs)
            except:
                print("lognormal fit failed for ",self.ID)
            # fit normal
            ax.plot(x, mlab.normpdf(x, mu, sigma), color='b', **kwargs)

        plt.hist(trs.noise[ivalid], bins=50, normed=True, facecolor='g')
        ax.set_xlabel('position error (m)')
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 1.0])
        ax.set_ylabel('frequency')

        return

    def get_mask(self, methods=None):
        """ get mask based on outlier flags """ 
        plot_mask = np.ones(self.ndetects, np.bool_)
        rec_tr = self.rec_track

        for nm,method in enumerate(methods):
            method_mask = rec_tr[method] == 1
            plot_mask = np.multiply(~method_mask, plot_mask)

        return plot_mask

    def plot_detects(self, ax, color_by=None, add_cbar=False, 
                     plot_smoothed=False, plot_outliers=False, 
                     plot_changepts=False, plot_changepts_fill=False,
                     mask=None, max_del_vel=1.0, **kwargs):
        """ draw positions and lines between positions on map plot """

        lines = []
        if mask is None:
            rec_tr = self.rec_track
        else:
            rec_tr = self.rec_track[mask]
        ndetects = len(rec_tr)
        for nd in range(ndetects-1):
            tr = rec_tr[nd]
            xy1 = [tr.X,tr.Y]
            tr = rec_tr[nd+1]
            xy2 = [tr.X,tr.Y]
            lines.append([xy1, xy2])
        time_from_entry = rec_tr.Sec - self.t_entry + 1
        log_age = np.log10(time_from_entry)
        if color_by == 'age':
            kwargs['array'] = np.asarray(log_age)
            kwargs['cmap'] = cm_age
        kwargs['linewidths'] = (0.8)
        tr_lines = LineCollection(lines, **kwargs)
        if color_by not in ['daynight','routesel']:
            ax.add_collection(tr_lines)
        elif color_by == 'routesel':
            self.identify_route_selection(mask)
        # add raw position dots to vertices of lines

        clims = [ticks[0],ticks[-1]]
        tr_lines.set_clim(clims)
        if add_cbar:
            label = 'Time from Entry (seconds)'
            c1 = plt.gcf().colorbar(tr_lines)
            c1.set_label(label)
            c1.set_ticks(ticks)
            c1.set_ticklabels(tick_labels)
        # plot flagged positions 
        if plot_outliers:
            kwargs['linewidths'] = (0.2)
            kwargs['linestyle'] = ('--')
            rec_tr_all = self.rec_track
            if mask is not None: # plot thin lines to outliers
                all_lines = []
                for nd in range(self.ndetects-1):
                    tr = rec_tr_all[nd]
                    xy1 = [tr.X,tr.Y]
                    tr = rec_tr_all[nd+1]
                    xy2 = [tr.X,tr.Y]
                    all_lines.append([xy1, xy2])
                if color_by == 'age':
                    time_from_entry = rec_tr_all.Sec - self.t_entry + 1
                    log_age_all = np.log10(time_from_entry)
                    kwargs['array'] = np.asarray(log_age_all)
                tr_lines_all = LineCollection(all_lines, **kwargs)
                if color_by in ['daynight','routesel']:
                    ax.add_collection(tr_lines_all)
                tr_lines_all.set_clim(clims)
                tr_lines_all.set_zorder(1)
            # plot flagged outliers
            for nm, method in enumerate(self.outlier_methods):
                omarker = outlier_marker[method]
                ocolor = outlier_color[method]
                if method == 'Dry':
                    color = ocolor
                else:
                    color = "None"
                flagged = np.where(rec_tr_all[method] == 1)[0]
                ax.scatter(rec_tr_all.X[flagged], rec_tr_all.Y[flagged], 
                           marker=omarker, edgecolor=ocolor, 
                           c=color, s=10.0, zorder=8)
        if color_by == 'age':
            pos = ax.scatter(rec_tr.X, rec_tr.Y, marker='.', s=2.6, 
                             cmap=cm_age, vmin=ticks[0], vmax=ticks[1])
        elif color_by == 'daynight':
            i = self.mnames.index('daytime_entry')
            day = self.metrics[i]
            if day:
                colr = 'r'
            else:
                colr = 'k'
            pos = ax.scatter(rec_tr.X, rec_tr.Y, marker='.', s=2.6, c=colr)
        elif color_by == 'routesel':
            if self.route == 'Old':
                colr = 'r'
            elif self.route == 'SJ':
                colr = 'g'
            else:
                colr = 'gold'
            pos = ax.scatter(rec_tr.X, rec_tr.Y, marker='.', s=2.6, c=colr)

        if plot_smoothed: # plot smoothed positions on top
            #trs = self.rec_smooth_pos
            trs = self.rec_smooth_fill
            ax.scatter(trs.X, trs.Y, marker='o', color='darkviolet', s=0.8,
                       zorder=9)

        if plot_changepts: # assumes smoothed position record is available
            trs = self.rec_smooth_pos
            #turn_angle = self.turn_angle(rec_track = trs)
            #turn_angle = trs.turn_angle
            mask = trs.change_pt_flag1
            ax.scatter(trs.X[mask], trs.Y[mask], marker='^', 
                       c='None',edgecolor='pink', s=8.0, zorder=9)
#                      c=cm_red(turn_angle[mask1]), s=8.0, zorder=9)
            mask = trs.change_pt_flag2
            ax.scatter(trs.X[mask], trs.Y[mask], marker='^', 
                       c='None',edgecolor='salmon', s=16.0, zorder=9)
#                      c=cm_red(turn_angle[mask2]), s=4.0, zorder=9)
            mask = trs.change_pt_flag3
            ax.scatter(trs.X[mask], trs.Y[mask], marker='^', 
                       c='None',edgecolor='r', s=32.0, zorder=9)
        if plot_changepts_fill: # assumes smoothed position record is available
            trs = self.rec_smooth_fill
            mask = trs.change_pt
            ax.scatter(trs.X[mask], trs.Y[mask], marker='^', 
                       c='None',edgecolor='r', s=32.0, zorder=9)
            # overwrite smoothed points using p_stat colorbar
            ps = ax.scatter(trs.X, trs.Y, marker='.', c=trs.p_stat, 
                            vmin=0, vmax=0.2, cmap=cm_red_r, s=1.0, zorder=9)
            cbar_ps = plt.gcf().colorbar(ps)
            cbar_ps.set_label('p statistic')
            cbar_ps.set_ticks([0,0.2])
            c1.set_ticklabels(tick_labels)



 
