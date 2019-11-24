"""
Investigate the potential for looking holistically at "all" detections
together, via a Bayesian inference approach.

Slimmed down from bayes_processing.py
"""

import pandas as pd
import numpy as np
import datetime
import logging as log
from stompy import filters
import xarray as xr
import six
import matplotlib.pyplot as plt
from matplotlib import collections
from scipy.optimize import fmin
from stompy import utils, memoize

import parse_tek as pt
six.moves.reload_module(pt)


##
AM1="2019 Data/HOR_Flow_TekDL_2019/AM1_186002/am18-6002190531229.DET"
AM2="2019 Data/HOR_Flow_TekDL_2019/AM2_186006/20190517/186006_190222_130811_P/186006_190222_130811_P.DET"
AM3="2019 Data/HOR_Flow_TekDL_2019/AM3_186003/20190517/186003_190217_125944_P/186003_190217_125944_P.DET"
AM4="2019 Data/HOR_Flow_TekDL_2019/AM4_186008/20190517/186008_190517_130946_P/186008_190517_130946_P.DET"
AM6="2019 Data/HOR_Flow_TekDL_2019/AM6_186005/20190517/186005_190224_141058_P/am18-6005190481424.DET"
AM7="2019 Data/HOR_Flow_TekDL_2019/AM7_186007/20190517/186007_190217_135133_P/am18-6007190481351.DET"
AM8="2019 Data/HOR_Flow_TekDL_2019/AM8_186009/_186009/186009_190222_132321_F/186009_190222_132321_F.DET"
AM9="2019 Data/HOR_Flow_TekDL_2019/AM9_186004/20190517/186004_190311_144244_P/am18-6004190440924.DET"

SM1="2019 Data/HOR_Flow_TekDL_2019/SM1_187023/20190517/187023_190416_050506_P/sm18-7023190711049.DET"
SM2="2019 Data/HOR_Flow_TekDL_2019/SM2_187013/2018_7013/187013_190516_143749_P.DET"
SM3="2019 Data/HOR_Flow_TekDL_2019/SM3_187026/20190516/187026_190301_124447_P/sm18-7026190421300.DET"
SM4="2019 Data/HOR_Flow_TekDL_2019/SM4_187020/20190516/187020_190301_123034_P/sm18-7020190421306.DET"
SM7="2019 Data/HOR_Flow_TekDL_2019/SM7_187017/20190516/187017_190225_154939_P/sm18-7017190391457.DET"
SM8="2019 Data/HOR_Flow_TekDL_2019/SM8_187028/20190516/187028_190216_175239_P/sm18-7028190421324.DET"
##

all_receiver_fns=[
    ('AM1',AM1),
    ('AM2',AM2),
    ('AM3',AM3),
    ('AM4',AM4),
    ('AM6',AM6),
    ('AM7',AM7),
    ('AM8',AM8),
    ('AM9',AM9),
    ('SM1',SM1),
    ('SM2',SM2),
    ('SM3',SM3),
    ('SM4',SM4),
    ('SM7',SM7),
    ('SM8',SM8)
]

all_detects=[pt.parse_tek(fn,name=name) for name,fn in utils.progress(all_receiver_fns)]
##

all_detects_nomp=[pt.remove_multipath(d) for d in all_detects]

##

# First cut: 
# Limit to 1 hour of detections. 
# Try to get all receivers sync'd to the point that pings line up

t_clip=[np.datetime64("2019-03-25 00:00"),
        np.datetime64("2019-03-25 01:00")]

clipped_detects=[ d.isel( index=(d.time.values>=t_clip[0]) & (d.time.values<t_clip[-1]))
                  for d in all_detects_nomp]

##

# Get that into one nice big table.
# For the moment, just care about time, tag and name.

all_rxs=xr.concat(clipped_detects,dim='index')

# before adding rx in
all_rx_df=all_rxs.to_dataframe()

# map receivers to a number 
all_rxs['names']=('rx',),np.unique(all_rxs.name.values)

names=all_rxs.names.values


all_rxs['rx']=('index',),np.searchsorted( names, all_rxs.name.values)

all_tags=np.unique( all_rxs.tag.values )


##

# 3388 detections over this 1 hour.

# Defining the parameters:

# for the moment, times are adjusted independently for each station.
# each station has a piecewise linear time offset
# start with one extra degree of freedom.

T0=np.datetime64('2019-02-01 00:00')
def to_tnum(x):
    """
    Local mapping of datetime64 to application specific time value.
    Currently floating point seconds since 2019-02-01, but trying
    to abstract the details away
    """
    return (x-T0)/np.timedelta64(1,'ns') * 1e-9

for d in clipped_detects:
    d['tnum']=('index',),to_tnum(d['time'].values)

all_rx_df['tnum']=to_tnum(all_rx_df['time'].values)

rx_tnums=[ d.tnum.values for d in clipped_detects]
rx_tags =[ d.tag.values for d in clipped_detects]

##

# Try more of a sweep approach
# Directly match pings, and continually check to see if
# the resulting solution is possible.

# For each receiver, I have tnums and tag ids:
#   rx_tnums=[ d.tnum.values for d in clipped_detects]
#   rx_tags =[ d.tag.values for d in clipped_detects]

# Each 'solution' is built incrementally, matching
# pings between receivers.
# If at some point the solution is not possible, then
# it is abandoned, and the next solution explored.

# Start with tag that has the most receptions?
# Can narrow the dataset some by ignoring tags
# that were seen by only 1 receiver.

tag_rx_count=all_rx_df.groupby(['tag','name']).size().groupby('tag').size().sort_values(ascending=False)

# 15 tags were seen at more than 1 rx.
multi_rx_tags= tag_rx_count[ tag_rx_count>1 ].index.values

grouped=all_rx_df.loc[:,['tag','name','id','tnum']].sort_values(['tag','name']).set_index(['tag','name'])

for tag,rx_count in tag_rx_count.iteritems():
    if rx_count<2: break
    
    break

print(f"Tag {tag} was seen by {rx_count} receivers")

# Here I want to line up detections for this single tag

# but it's going to suffer a lot of local minima.
#

##

# tag_rx_tnums=[]
# tag_rx_counts=[]
# 
# for rx_i in range(len(names)):
#     rx_tag_sel=rx_tags[rx_i]==tag
#     if not np.any(rx_tag_sel): continue
#     tnums=rx_tnums[rx_i][rx_tag_sel]
#     tag_rx_tnums.append(tnums)
#     tag_rx_counts.append(len(tnums))
    
##

# for enumerate_groups tag must be numeric
def tag_to_num(t):
    return np.searchsorted(all_tags,t)

beacon_tagns=tag_to_num(all_rx_df.beacon_id.unique())

tagn_interval={}
for t in all_tags:
    tagn=tag_to_num(t)
    if tagn in beacon_tagns:
        # beacon tags every 30 seconds
        tagn_interval[tagn]=30.0
    else:
        # fish tags every 5 seconds
        tagn_interval[tagn]=5.0

##

# This pair works.. how about another
#  id_a=3 # AM4
#  id_b=12 # SM7

# This works - finds two non-empty solutions, with one clearly better
#  id_a=2 
#  id_b=12 # SM7

# empty.  37 is the only tag they have in common.  That's FF01,
# a has only one detect, and it indeed does not fall within max_shift.
# id_a=1
# id_b=12

# Good
# id_a=0
# id_b=1


tnums_a=rx_tnums[id_a]
tnums_b=rx_tnums[id_b]

tags_a=tag_to_num(rx_tags[id_a])
tags_b=tag_to_num(rx_tags[id_b])

if 0: # Testing -- revert to considering only one tag at a time
    # 'C5EA', 'FF01', 'FF02', 'FF03', 'FF05', 'FF07', 'FF08', 'FF17',
    # 'FF20', 'FF26', 'FF28'

    # only B saw BB8A. saw it 62 times.  and including that
    # breaks things.
    # FF20 and BB8A are close to a minimum example of it breaking
    allow_tags=['C5EA','FF01','FF02','FF03','FF05','FF07',
                'FF08', 'FF17',
                'FF20',
                'FF26','FF28',
                # That's everything in A, but B has others:
                '95F9',
                'BB8A','C55C' , 'FF04'
    ]
    
    a_sel=np.zeros(len(tags_a),np.bool8)
    b_sel=np.zeros(len(tags_b),np.bool8)
    for t in allow_tags:
        a_sel=a_sel|(tags_a==tag_to_num(t))
        b_sel=b_sel|(tags_b==tag_to_num(t))
    tnums_a=tnums_a[a_sel]
    tnums_b=tnums_b[b_sel]
    tags_a=tags_a[a_sel]
    tags_b=tags_b[b_sel]
    
print(f"Aligning {len(tnums_a)} hits at {names[id_a]} with {len(tnums_b)} at {names[id_b]}")

# ---
# where in the sequence we are

# decision tree
#  (i) match the next two tnums
#  (ii) skip the next tnum in id_a,
#  (iii) skip the next tnum in id_b

# when including all tags, only got single matches on
# C5EA.
max_shift=20.0
max_drift=0.005 # even if final drift is small, intermediate states may have higher values.
max_delta=0.500

def test_matches(matches,next_a,next_b,verbose=False):
    """
    return True if the matches so far, having considered
    up to next_a and next_b, are possible.
    """
    # special case -- when nothing matches, at least disallow
    # the search from getting too far ahead on one side or the
    # other. Can't force exact chronological order.
    # proceed in raw chronological order
    if len(matches)==0:
        # is a too far ahead of b?
        if next_a>0 and next_b<len(tnums_b) and tnums_a[next_a-1]>tnums_b[next_b] + max_shift:
            # we already looked at an a that comes after the next b.
            return False
        if next_b>0 and next_a<len(tnums_a) and tnums_b[next_b-1]>tnums_a[next_a] + max_shift:
            # already looked at a b that comes after the next a
            return False
        # No matches to consider, and next_a and next_b are okay, so carry on.
        return True
    
    amatches=np.array(matches)

    # Keep us honest, that the caller didn't try to sneak a bad match in.
    assert np.all( tags_a[amatches[:,0]]==tags_b[amatches[:,1]] )

    # evaluate whether a matched sequence is within tolerance
    a_times=tnums_a[amatches[:,0]]
    b_times=tnums_b[amatches[:,1]]

    if len(a_times)>1:
        mb=np.polyfit(a_times,b_times,1)
    else:
        mb=[1.0,b_times[0]-a_times[0]]

    if verbose:
        print(f"Drift is {1-mb[0]:.4e}")
        print(f"Shift is {np.mean(a_times-b_times):.3f}")
        print("Max error is %.3f"%np.abs(np.polyval(mb,a_times)-b_times).max())
        
    if not np.abs(np.log10(mb[0]))<np.log10(1+max_drift):
        return False # drift is too much

    # the intercept is the extrapolated time at tnum epoch.
    if not np.abs(np.mean(a_times-b_times))<max_shift:
        return False # shift is too much

    max_error=np.abs(np.polyval(mb,a_times)-b_times).max()
    if not max_error < max_delta:
        return False # resulting travel times are too much.

    # to consider: instead of just going through next_a and next_b,
    #  this could go through the later of the adjusted times
    dtype=[('tnum',np.float64),
           ('matched',np.bool8),
           ('tag',np.int32)]
    a_full=np.zeros(next_a,dtype=dtype)
    b_full=np.zeros(next_b,dtype=dtype)
    
    a_full['tnum']=np.polyval(mb,tnums_a[:next_a])
    b_full['tnum']=tnums_b[:next_b]
    a_full['matched'][amatches[:,0]]=True
    b_full['matched'][amatches[:,1]]=True
    a_full['tag']=tags_a[:next_a]
    b_full['tag']=tags_b[:next_b]

    combined=np.concatenate([a_full,b_full])
    order=np.argsort(combined['tnum'])
    combined=combined[order]

    # This part has to be done per tag:
    for tagn,idxs in utils.enumerate_groups(combined['tag']):
        if len(idxs)<2: continue
        deltas=np.diff( combined['tnum'][idxs] )
        matched=np.minimum( combined['matched'][idxs][:-1],
                            combined['matched'][idxs][1:] )

        if verbose:
            if np.any(~matched):
                min_delta=deltas[~matched].min()
            else:
                min_delta=np.nan
            print(f"[%s] %3d total detect. Min delta %.3f vs. interval %.3f for tag %s"%(all_tags[tagn],
                                                                                         len(idxs),
                                                                                         min_delta,
                                                                                         tagn_interval[tagn],
                                                                                         tagn))
        # evaluate all, sort, look at diff(time)
        if np.any( deltas[~matched] < 0.8*tagn_interval[tagn]):
            # the 0.8 is some slop. 
            return False
    return True

full_matches=[]

# for the purpose of matching just two sequences, shift
# and drift are two numbers.

def match_remainder(next_a,next_b,matches,visited=None):
    """
    Recursively enumerate allowable sequence matches.
    next_a: index into tnums_a of the next unmatched detection
    next_b: ... tnums_b
    matches: list of index pairs ala next_a/b that have been matched.
    """
    if visited is None:
        visited={}
        
    # Check to see if this has already been tried
    key=(next_a,next_b,tuple(matches))
    if key in visited: return
    visited[key]=1

    if not test_matches(matches,next_a,next_b):
        return
        
    if (next_a==len(tnums_a)) and (next_b==len(tnums_b)):
        # termination condition
        full_matches.append(matches)
        print("MATCH")
        return
    elif (next_a==len(tnums_a)):
        # finish out b
        match_remainder(next_a,len(tnums_b),matches,visited)
        return
    elif (next_b==len(tnums_b)):
        # finish out a
        match_remainder(len(tnums_a),next_b,matches,visited)
        return

    # is it possible to declare the next two a match?
    if (tags_a[next_a]==tags_b[next_b]):
        time_diff=tnums_a[next_a]-tnums_b[next_b]
        if (np.abs(time_diff)<max_shift):
            # sure, try a match
            match_remainder(next_a+1,next_b+1,matches + [(next_a,next_b)],visited)
        else:
            # Tags match, but time diff is too big for max_shift
            pass

    if tnums_a[next_a]<tnums_b[next_b] + max_shift:
        # drop which ever one is earlier. doesn't work in general, but I'm not
        # explicitly including shifts yet
        match_remainder(next_a+1,next_b,matches,visited)
    if tnums_b[next_b]<tnums_a[next_a] + max_shift:
        match_remainder(next_a,next_b+1,matches,visited)

    
match_remainder(0,0,[])

print(f"Found {len(full_matches)} potential sets of matches")

# 100ms to find two good sets of matches, and an empty.
# The max allowed shift and drift could, in theory, rule out the
# empty sets, if there is no way to align the sequences within those
# bounds and not have things that are too close.
# but currently I don't have a good way to 

# In this particular case, the two non-empty matches are
# both very good.

for i,matches in enumerate(full_matches):
    print()
    print(f"---- Match {i}  ({len(matches)} pairs) ---")
    test_matches(matches,len(tnums_a),len(tnums_b),verbose=True)
##

matches=full_matches[0]
## 
# show those matches:
amatches=np.array(full_matches[0])
apnts=np.array( [tnums_a[amatches[:,0]],
                 id_a*np.ones(len(amatches))])
bpnts=np.array( [tnums_b[amatches[:,1]],
                 id_b*np.ones(len(amatches))])

ab_segs=np.array([apnts,bpnts]).transpose(2,0,1)

##

fig=plt.figure(1)
fig.clf()
ax=fig.add_subplot(1,1,1)

for ii,i in enumerate([id_a,id_b]):
    ax.plot( tag_rx_tnums[i],i*np.ones(len(tag_rx_tnums[i])),
             'o')
ax.add_collection( collections.LineCollection(ab_segs) )
