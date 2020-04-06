import numpy as np
import os, shutil
import pandas as pd

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
        val=df[col][ df[col].notnull()].values[0]
        if isinstance(val,pd.DataFrame):
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
    
    track['ground_u']=np.r_[dx/dt,np.nan]
    track['ground_v']=np.r_[dy/dt,np.nan]
    track['swim_u']=track['ground_u'] - track[model_u]
    track['swim_v']=track['ground_v'] - track[model_v]

    # headings in radians, ccw from eastward/+x
    track['model_hdg'] = np.arctan2(track[model_v],track[model_u])
    track['swim_hdg'] = np.arctan2(track['swim_v'],track['swim_u'])

    # swim heading ccw from hydro heading,
    swim_hdg_rel=(track['swim_hdg']-track['model_hdg'])
    # wrap to [-pi,pi]
    track['swim_hdg_rel']=(swim_hdg_rel+np.pi)%(2*np.pi) - np.pi

    track['swim_urel'],track['swim_vrel']=rot(track['model_hdg'],
                                              track['swim_u'],
                                              track['swim_v'])
