"""
Generate some summary numbers from the ping processing for
the swimspeed manuscript
"""
import os
import numpy as np
import pandas as pd
from shapely import geometry
from stompy.grid import unstructured_grid
from stompy.plot import plot_wkb

import glob
import six
## 
# From sync_pings
data_dirs=[
        'yaps/full/2018/20180313T1900-20180316T0152',
        'yaps/full/2018/20180316T0152-20180321T0003',
        'yaps/full/2018/20180321T0003-20180326T0000',
        'yaps/full/2018/20180326T0000-20180401T0000',
        'yaps/full/2018/20180401T0000-20180406T0000',
        'yaps/full/2018/20180406T0000-20180410T0000',
        'yaps/full/2018/20180410T0000-20180415T0100'
]

dfs=[ pd.read_csv(os.path.join(data_dir,'all_detections.csv'))
      for data_dir in data_dirs]

##

df_tot=pd.concat(dfs)

print("Total detections (non-multipath, clipped to time in the water): ",len(df_tot))
# Total detections:  2765610
# That's about half the number from just counting lines in DET files, with
# the discrepancy due mostly to out-of-water pings, but also some multipath.

## 

# Remove sync pings:
hydros=pd.read_csv("../circulation/hydros.csv")
hydro_tags=hydros['sync_tag'].values

##

df_nonhydro=df_tot[ ~df_tot.tag.isin(hydro_tags).values ]

print(f"Detections from non-sync tags: {len(df_nonhydro)}")
# Detections from non-sync tags: 183951

##

# And further limit to known fish tags.
fish_tags_df=pd.read_excel("../circulation/2018_Data/2018FriantTaggingCombined.xlsx")
fish_tags=fish_tags_df['TagID_Hex'].values

# Tracking down the issue with 772A and 82A4: they are both listed in the upper
# and lower releases.
# Not enough information to 100% confirm, but 7B51 and 7DAB are not in the
# the xlsx file, yet have very realistic, fish-y tracks.  So maybe the duplicate
# 772A and 82A4 are in fact 7B51 and 7DAB.
# I've arbitrarily replace duplicates in the xlsx with 7B51 and 7DAB (and
# added a comment to that effect to the xlsx cells).

if 0:
    # This has some tags that weren't fish (e.g. drifters)
    taglist=pd.read_csv("../circulation/2018_Data/2018FriantTagList.csv")
    # np.setdiff1d( taglist.TagID_Hex.values, fish_tags_df.TagID_Hex.values)
    # These are tag-ids in the csv that do not show up in the xlsx.
    #   '6EA2', '6FD5', '722A', '726D', '7295', '729B', '72A4', '72BA',
    #   '72D4', '7354', '7357', '746A', '746B', '74A5', '74AC', '7505',
    #   '7513', '752D', '752F', '753B', '7541', '756F', '7592', '7594',
    #   '75B2', '75B9', '75BB', '762B', '7649', '7669', '76A2', '76E9',
    #   '7757', '7959', '7A4D', '7A59', '7A75', '7A9B', '7AD2', '7B4A',
    #   '7B51', '7B52', '7B5B', '7B95', '7D4A', '7DAB', '8254', '8256',
    #   '82A5', '82AA', '82CA', '82DA', '8352', '8355', 'unknown'

    # Anybody one digit off from 772A or 82A4?
    # '722A'
    # '72A4'
    # '8254'
    # '82A5'
    # '82AA'
    # Do any of those have extra info in the csv? nope.

    # And do they have any detects? nope.
    for maybe_fish in ['722A', '72A4', '8254', '82A5', '82AA']:
        print(maybe_fish)
        print( df_nonhydro[ df_nonhydro.tag==maybe_fish ] )

    ## check the whole list then.
    for maybe_fish in np.setdiff1d( taglist.TagID_Hex.values,
                                    fish_tags_df.TagID_Hex.values):
        sel=(df_nonhydro.tag==maybe_fish)
        hits=sel.sum()
        if hits:
            print(maybe_fish,hits)
            times=df_nonhydro[sel].epo.values
            print(utils.unix_to_dt64(times.min()),
                  utils.unix_to_dt64(times.max()))

    # Potential tags and corresponding ping count in df_nonhydro,
    # dropping two that are noted as drifters.
    # 7B51 127 -- does come up with a track -- could be real.
    #             comes through 3/25.
    # 7DAB 142 -- has a track -- looks real
    #             comes through 3/28

##     
df_fish=df_nonhydro[ df_nonhydro.tag.isin(fish_tags) ]

print(f"Detections from known fish tags: {len(df_fish)}")

# Detections from known fish tags: 148260

print("Unique fish tags received: ",len(df_fish.tag.unique()))
# Unique fish tags received:  348

## Label those by release, and write out to csv for use in timeline
# figure

grps=df_fish.groupby('tag')
df_detects=grps.first()
del df_detects['serial']
del df_detects['frac']
df_detects['tag']=df_detects.index

upper=fish_tags_df['Release Date'] < np.datetime64("2018-03-04")
lower=~upper # some NaT, mostly 2018-03-15
fish_tags_df['release']=np.where(upper,"upper","lower")

df_detect_w_release=df_detects.merge( fish_tags_df[ ['TagID_Hex','release']],
                                      left_index=True,right_on='TagID_Hex',
                                      how='left')
df_detect_w_release=df_detect_w_release.set_index('tag')
df_detect_w_release.to_csv("tag-detections_w_release.csv")


##

g=unstructured_grid.UnstructuredGrid.read_ugrid("../../model/grid/snubby_junction/snubby-08-edit60.nc")
g_poly=g.boundary_polygon()

# comparison to tekno solutions:
tekno=pd.read_csv("../tags/cleaned_half meter.csv")




# 'raw' output from YAPS;
raw_csv_patt='yaps/full/v06/track*.csv'
raw_csvs=glob.glob(raw_csv_patt)

dfs=[]
for fn in raw_csvs:
    df=pd.read_csv(fn)
    del df['Unnamed: 0']
    basename=os.path.basename(fn)
    df['id']=basename.replace('track-','').replace('.csv','')
    dfs.append(df)
    
yaps=pd.concat(dfs)
yaps3=yaps[ yaps.num_rx>=3 ].copy()

yaps3['in_poly']=[ g_poly.contains( geometry.Point(r.x,r.y) )
                   for idx,r in yaps3.iterrows() ]

tekno['in_poly']=[ g_poly.contains( geometry.Point(r.X_UTM,r.Y_UTM) )
                   for idx,r in tekno.iterrows() ]


## 
# Trim to common periods:
t_min=max( tekno.Epoch_Sec.min(),
           yaps3.tnum.min() )
t_max=min( tekno.Epoch_Sec.max(),
           yaps3.tnum.max() )

yaps3_clip=yaps3[ (yaps3.tnum>=t_min) & (yaps3.tnum<=t_max) ]
tekno_clip=tekno[ (tekno.Epoch_Sec>=t_min) & (tekno.Epoch_Sec<=t_max) ]

##

yaps_in=yaps3_clip.in_poly.sum()
yaps_tot=len(yaps3_clip)
print(f"{yaps_in}/{yaps_tot} ({100.0*yaps_in/yaps_tot:.2f}% 3+ rx yaps solutions are within the domain")
print(f"{yaps_tot-yaps_in}/{yaps_tot} ({100.0*(1.0-yaps_in/yaps_tot):.2f}% 3+ rx yaps solutions are outside the domain")
tek_in=tekno_clip.in_poly.sum()
tek_tot=len(tekno_clip)
print(f"{tek_in}/{tek_tot} ({100.0*tek_in/tek_tot:.2f}% Teknologic solutions are within the domain")
print(f"{tek_tot-tek_in}/{tek_tot} ({100.0*(1.0-tek_in/tek_tot):.2f}% Teknologic solutions are outside the domain")

##

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

sel=tekno_clip.in_poly
ax.plot(tekno_clip.loc[sel,'X_UTM'],tekno_clip.loc[sel,'Y_UTM'],'k.')
sel=~sel
ax.plot(tekno_clip.loc[sel,'X_UTM'],tekno_clip.loc[sel,'Y_UTM'],'k+')

sel=yaps3_clip.in_poly
ax.plot(yaps3_clip.loc[sel,'x'],yaps3_clip.loc[sel,'y'],'r.')
sel=~sel
ax.plot(yaps3_clip.loc[sel,'x'],yaps3_clip.loc[sel,'y'],'r+')

plot_wkb.plot_wkb(g_poly,ax=ax,zorder=-2,alpha=0.3)
ax.axis('equal')
