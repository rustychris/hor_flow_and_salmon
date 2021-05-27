import numpy as np
import glob
import os, shutil
import pandas as pd
from shapely import geometry
from stompy.spatial import wkb2shp
from stompy import utils
from matplotlib.path import Path

from stompy.plot import plot_wkb
from stompy.grid import unstructured_grid


def dump_to_folder(df,path,clean=True):
    """
    Write a dataframe to csv, and any dataframe-valued fields
    are written as separate csvs, indexed by the field name
    """
    if clean and os.path.exists(path):
        for fn in glob.glob(os.path.join(path,'*.csv')):
            os.unlink(fn)
        
    if not os.path.exists(path):
        os.makedirs(path)

    cols=df.columns.values
    df_cols=[]
    non_df_cols=[]
    for col in cols:
        vals=df[col][ df[col].notnull()].values
        if len(vals) and isinstance(vals[0],pd.DataFrame):
            df_cols.append(col)
        else:
            non_df_cols.append(col)

    df_nodf=df.loc[:,non_df_cols].copy()
    
    for df_col in df_cols:
        fn_fld=df_col+"_fn"
        df_nodf[fn_fld]=["%s-%s.csv"%(df_col,idx) for idx in df_nodf.index.values]

        for idx,row in df.iterrows():
            if row[df_col].values is None: continue
            fn=os.path.join(path,df_nodf.loc[idx,fn_fld])
            row[df_col].to_csv(fn,index=False)

    df_nodf.to_csv(os.path.join(path,'master.csv'))

def read_from_folder(path):
    """
    reverse of dump_to_folder.
    """
    df_out=pd.read_csv(os.path.join(path,'master.csv'),index_col=0)

    fn_cols=[col for col in df_out.columns.values if col.endswith('_fn')]

    def ingest_csv(fn):
        return pd.read_csv(os.path.join(path,fn))
    
    for fn_col in fn_cols:
        df_col=fn_col.replace('_fn','')
        df_out[df_col]=df_out[fn_col].apply(ingest_csv)

        del df_out[fn_col]
    return df_out

def rot(theta,srcx,srcy):
    """
    rotate the coordinate frame by theta.
    points given by (srcx,srcy) 
    theta: angle of the new +x axis in the original reference frame, in 
    radians CCW of +x.
    """
    dstx=srcx*np.cos(theta) + srcy*np.sin(theta)
    dsty=-srcx*np.sin(theta) + srcy*np.cos(theta)
    return dstx,dsty

def calc_velocities(track,model_u='model_u_surf',model_v='model_v_surf'):
    """
    Add columns for derived velocity quantities based on velocity over
    ground and hydrodynamic velocity (as specified by fields noted in 
    model_u and model_v).

    Adds: 
    ground_{u,v}: velocity over ground, E/N coordinates
    swim_{u,v}: swim velocity, E/N coordinates
    model_hdg: heading of model flow, ccw from eastward
    swim_hdg: heading of swim vector, ccw from eastward
    swim_hdg_rel: heading of swim vector, ccw from flow heading
    swimg_{u,v}rel: swim velocity rotated to urel==downstream, vrel=river left

    Updates the track in place.
    """
    dx=np.diff(track.x.values)
    dy=np.diff(track.y.values)
    dt=np.diff(track.tnum)

    track['vel_dt']=np.r_[dt,np.nan]
    track['ground_u']=np.r_[dx/dt,np.nan]
    track['ground_v']=np.r_[dy/dt,np.nan]
    track['swim_u']=track['ground_u'] - track[model_u]
    track['swim_v']=track['ground_v'] - track[model_v]

    # headings in radians, ccw from eastward/+x
    track['model_hdg'] = np.arctan2(track[model_v],track[model_u])
    track['swim_hdg']  = np.arctan2(track['swim_v'],track['swim_u'])
    track['ground_hdg']= np.arctan2(track['ground_v'],track['ground_u'])

    # swim heading ccw from hydro heading,
    swim_hdg_rel=(track['swim_hdg']-track['model_hdg'])
    # wrap to [-pi,pi]
    track['swim_hdg_rel']=(swim_hdg_rel+np.pi)%(2*np.pi) - np.pi

    track['swim_urel'],track['swim_vrel']=rot(track['model_hdg'],
                                              track['swim_u'],
                                              track['swim_v'])

    track['ground_urel'],track['ground_vrel']=rot(track['model_hdg'],
                                                  track['ground_u'],
                                                  track['ground_v'])

def dump_to_shp(df,shp_fn,**kwargs):
    geoms=[geometry.LineString( np.c_[rec['x'].values,
                                      rec['y'].values] )
           for rec in df['track'].values ]

    fields=dict()
    fields['index']=df.index.values
    for col in df.columns.values:
        col_safe=col[:10]
        if col!='track':
            fields[col_safe]=df[col].values
            
    wkb2shp.wkb2shp(shp_fn,geoms,
                    fields=fields,
                    **kwargs)

def gate_crossings(track,gates):
    """
    track: a dataframe with x and y columns giving track coordinates and a
     tnum column with numeric times

    gates: sequence of LineString geometries

    returns an array of crossings in chronological order.
    each crossing is stored as a tuple:
      (gate_index, time, left_to_right,x-coord,y-coord)
    
    gate_index is 0-based.  left_to_right is based on looking along the gate,
    from starting vertex to ending vertex, and is true if the track starts
    to the left and goes to the right.  
    """
    crossings=[] # (gate_idx,time,+1 for left-to-right)

    for gate_idx,rec in enumerate(gates):
        if 'levee' in rec['name']: continue
        gate_pnts=np.array(rec['geom'])

        pnts=np.c_[track.x,track.y]
        track_line=geometry.LineString(pnts)

        if not track_line.intersects(rec['geom']):
            continue

        # Loop through segments of the track and identify times of intersection, and
        # the sign of each intersection
        for i in range(len(pnts)-1):
            seg=geometry.LineString(pnts[i:i+2,:])
            for j in range(len(gate_pnts)-1):
                gate_seg=geometry.LineString(gate_pnts[j:j+2])

                isect=seg.intersection(gate_seg)
                if isect.type=='GeometryCollection':
                    assert len(isect)==0,"Thought this was going to be an empty collection"
                    continue
                assert isect.type=='Point'
                # find the time of sign of crossing.
                p=np.array(isect)
                alpha=utils.dist(p-pnts[i])/utils.dist(pnts[i+1]-pnts[i])
                assert alpha<=1.0
                t=(1-alpha)*track.tnum.values[i] + alpha*track.tnum.values[i+1]

                seg_vec=pnts[i+1]-pnts[i]
                gate_vec=gate_pnts[j+1]-gate_pnts[j]
                c=np.cross(seg_vec,gate_vec)
                assert c!=0,"Bad luck - colinear?!"
                crossings.append( [gate_idx,t,c>0,p[0],p[1]] )

    times=[c[1] for c in crossings]
    order=np.argsort(times)
    result=np.zeros( len(times), [ ('gate',np.int32),
                                   ('tnum',np.float64),
                                   ('l2r',np.bool8),
                                   ('x',np.float64),
                                   ('y',np.float64) ] )
    for i,src in enumerate(order):
        result['gate'][i] = crossings[src][0]
        result['tnum'][i] = crossings[src][1]
        result['l2r'][i]  = crossings[src][2]
        result['x'][i]    = crossings[src][3]
        result['y'][i]    = crossings[src][4]
        
    return result
    
def expand_to_regular(df,t,dt_nom=5.0,fields='all',fill=np.nan):
    """
    Infer missing time steps in data, replace with
    fill value, or if 'interp', use linear interpolation
    """
    if fields=='all':
        fields=df.columns.values
        
    t=np.asarray(t)
    di=np.round(np.diff(t)/dt_nom).astype(np.int32)
    assert di.min()>0
    n_expand = di.sum()+1
    tgts=np.r_[0,np.cumsum(di)]

    valid=np.zeros(n_expand,np.bool8)
    valid[tgts]=True

    out=pd.DataFrame()
    for f in fields:
        f_out=np.nan*np.zeros(n_expand,np.float64)
        f_out[valid]=df[f].values
        if fill=='interp':
            f_out[~valid] = np.interp(np.nonzero(~valid)[0],
                                      tgts,df[f].values)
        else:
            f_out[~valid]=fill
        out[f]=f_out
    return out

def set_direction_labels(ax):
    ax.set_ylabel('Swim speed (m/s)')
    ax.set_xlabel('Swim speed (m/s)')

    bbox=dict(facecolor='w',lw=0.5)
    common=dict(bbox=bbox,weight='bold',transform=ax.transAxes)

    ax.text( 0.5, 1.0, 'Downstream',va='top',ha='center', **common)
    ax.text( 0.5, 0.0, 'Upstream',  va='bottom',ha='center', **common)
    ax.text( 0.0, 0.5, 'River\nLeft',va='center',ha='left',**common)
    ax.text( 1.0, 0.5, 'River\nRight',  va='center',ha='right',**common)

def medfilt_weighted(data,N,weights=None,quantile=0.5):
    """
    Running median filter with weights for each data point
    data: np.ndarray shape [M]
    N: length of the window (weights are not accounted for in the window size)
    weights: np.ndarray shape [M]
    quantile: compute alternate percentile (0.5: median)
    """
    if weights is None:
        weights=np.ones(len(data))

    data=np.asarray(data)
    weights=np.asarray(weights)
    
    result=np.zeros_like(data)
    neg=N//2
    pos=N-neg
    for i in range(len(result)):
        # say N is 5
        # when i=0, we get [0,3]
        # when i=10, we get [8,13
        slc=slice( max(0,i-neg),
                   min(i+pos,len(data)) )
        subdata=data[slc]
        # Sort the data
        ind_sorted = np.argsort(subdata)
        sorted_data = subdata[ind_sorted]
        sorted_weights = weights[slc][ind_sorted]
        # Compute the auxiliary arrays
        Sn = np.cumsum(sorted_weights)
        # center those and normalize
        Pn = (Sn-0.5*sorted_weights)/Sn[-1]
        result[i]=np.interp(quantile, Pn, sorted_data)
    return result

def analysis_polygon():
    g=unstructured_grid.UnstructuredGrid.from_shp('track_gates-v00.shp',max_sides=50)
    g.make_cells_from_edges(max_sides=50)
    # Which cells do I want?
    # center above the junction
    line_sel=geometry.LineString( [(647386.0027503968, 4185726.9867515), (647229.4335734614, 4185864.215030108)])

    for c in g.select_cells_intersecting(line_sel,as_type='indices',invert=True):
        g.delete_cell(c)
    g.delete_orphan_edges()
    g.delete_orphan_nodes()

    clip_poly=g.boundary_polygon_by_union()
    return clip_poly

def analysis_path():
    clip_poly=analysis_polygon()
    return Path( np.array( clip_poly.exterior ) )

def clip_to_analysis_polygon(df,label,min_remaining=10):
    """
    df: DataFrame with tracks and other data
    label: the column label holding the track to be trimmed
    min_remaining: if fewer than this number of samples remain, remove
    the row from the dataframe.

    Modifies tracks and the dataframe in place.
    """
    # What about a track that leaves the region and then comes back?
    # Make a point to clip *after* calculating swim velocities, and then
    # just include all the points inside the analysis region, since the
    # motivation is just to exclude points where the swim speed is not
    # reliable.
    # Alternatively I could nan out swim velocities outside the polygon,
    # but that's going to require a tweak to every plot.

    clip_path=analysis_path()

    for k,track in df[label].iteritems():
        inside=clip_path.contains_points( track.loc[:, ['x','y']].values )

        track.drop(list(np.nonzero(~inside)[0]),
                   inplace=True,axis=0)

    lengths=[ len(t) for t in df[label].values ]
    too_short=np.array(lengths)<min_remaining
    df.drop( df[too_short].index, inplace=True)
    
    
    
