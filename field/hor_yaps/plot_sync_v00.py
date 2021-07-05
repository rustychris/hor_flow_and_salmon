"""
Ground truth a period of sync data.

There is some weirdness in some of the solutions.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
## 
data_dir="yaps/full/2020/20200409T1100-20200409T1700"

hydros=pd.read_csv(os.path.join(data_dir,'hydros.csv'))
det=pd.read_csv(os.path.join(data_dir,'all_detections.csv'))
##

# Limit this to the known sync tags by joining with hydro
# now idx is the 1-based hydro idx
sync_det=det.merge(hydros,left_on='tag',right_on='sync_tag')
sync_det['tag_idx']=sync_det['tag'].astype('category')
sync_det['rx']=sync_det['serial_x'].astype('category')

##

print(f"{len(hydros)} hydrophones")
print(f"{len(sync_det.tag.unique())} unique sync tags")

##

# How to plot this to see correlation?
# Start with hydrophones=color, y=tag, x=time
fig=plt.figure(1)
fig.clf()

plt.scatter( sync_det.epo, sync_det.tag_idx, 20, sync_det.rx.cat.codes,
             cmap='jet')

# 
mat=sync_det.groupby(['tag_idx','rx']).size().unstack('tag_idx')

# so SM10 and SM11 hear each other, but don't hear others very well
# HERE: any indications of offsets greater than 30s?

##

from stompy import utils
utils.path("../tags")
import prepare_pings

## 
# Back to what's in combine_for_yaps

# pretty slow...
pm=prepare_pings.ping_matcher_2020(split_on_clock_change=True)
pm.allow_dual_bad_pings=True
# full range that seems to have a chance of resolving tracks
# This almost certainly too early and too late.
breaks=[np.datetime64('2020-03-01T00:00:00'),
        # Make a nice manageable chunk in the middle for testing
        # But there were no fish in that chunk
        np.datetime64('2020-03-17T00:00:00'),
        np.datetime64('2020-03-17T06:00:00'),
        # So make a nice chunk that seems to have some fish:
        # Note that this slicing cannot at the moment be handled in yaps,
        # because there are clock resets that create multiple copies of
        # the same hydrophone, which angers yaps.
        np.datetime64('2020-04-09 11:00:00'),
        np.datetime64('2020-04-09 17:00:00'),
        # and generous end point
        np.datetime64('2020-05-20T00:00:00')]

# SM3 has no valid data - prepare_pings comments very few
# detections, no dbg file.
min_duration=np.timedelta64(1,'h')

nonshort=[]
for df in pm.all_detects:
    t_min=max(breaks[0],df.time.values.min())
    t_max=min(breaks[-1],df.time.values.max())
    if t_max-t_min>min_duration:
        nonshort.append(df)
    else:
        print(f"{df.name.item()} is too short: {(t_max-t_min)/np.timedelta64(1,'h'):.2f}h")
print(f"{len(nonshort)} of {len(pm.all_detects)} hydrophone records were long enough")

pm.all_detects=nonshort
name_blacklist=[]

##
pm_rm=pm.remove_multipath()
pm_clip=pm_rm.clip_time(breaks[3:5])
pm_clip.remove_empty()
# Down to 11.  good!
##

# This finished in maybe 2 minutes
merged=pm_clip.match_all_by_similarity()

##

# plot these
pm_clip.summarize_clocks(merged)

plt.figure(2).clf()
fig,axs=plt.subplots(1,4,num=2)

for i,field in enumerate(['offset','drift','n_common','rmse']):
    img=axs[i].imshow(merged[field],cmap='jet')
    plt.colorbar(img,orientation='horizontal',ax=axs[i])
    axs[i].set_title(field)
    

    
# OK: no offsets are too large.  rmse is not too crazy.
# no gains to be had with ping_matcher
