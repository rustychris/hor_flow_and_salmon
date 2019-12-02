"""
Investigate the potential for looking holistically at "all" detections
together, via a Bayesian inference approach.

Slimmed down from bayes_processing.py

Refactored to keep more metadata from bayes_processing01.py
"""

import sys
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
# Limit to 1 hour of detections. 3388 detections over this 1 hour.

t_clip=[np.datetime64("2019-03-25 00:00"),
        np.datetime64("2019-03-25 01:00")]

clipped_detects=[ d.isel( index=(d.time.values>=t_clip[0]) & (d.time.values<t_clip[-1]))
                  for d in all_detects_nomp]

##

# scan to get complete list of receivers and tags
all_rx_names=[ ds.name.item() for ds in clipped_detects]
all_tags=np.unique( np.concatenate( [np.unique(ds.tag.values) for ds in clipped_detects] ) )

# for enumerate_groups tag must be numeric
def tag_to_num(t):
    return np.searchsorted(all_tags,t)

beacon_tagns=tag_to_num( [ds.beacon_id for ds in clipped_detects] )

##
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

##


# Core of the matching algorithm:
#   test_matches(), match_remainder()
#   These should be kept to simple (python and numpy types)
#   inputs, and left as standalone methods.  At some point
#   they may be optimized via numba and that will be easier
#   with simple inputs.
def test_matches(next_a,next_b,matches,
                 tnums_a,tnums_b,tags_a,tags_b,
                 verbose=False,
                 max_shift=20.0,max_drift=0.005,max_delta=0.500):
    """
    return True if the matches so far, having considered
    up to next_a and next_b, are possible.

    Search state:
    matches: a list of tuples [ (a_idx,b_idx), ... ]
      noting which pings are potentially the same.

    next_a,next_b: index into tnums_a, tnums_b for the next detection
     not yet considered.
    
    tnums_a,tnums_b: timestamps for the pings.
    tags_a,tags_b:  tag ids (numbers, not strings) for the pings.

    max_shift: two clocks can never be more than this many seconds 
    apart, including both shift and drift. This is not 100% enforced,
    but in some cases this condition is used to prune the search.

    max_drift: unitless drift allowed. 0.001 would be a drift of 1ppt,
    such that every 1000 seconds, clock a is allowed to lose or gain 
    1 second relative to clock b.

    max_delta: the limit on how far apart detections can be and still be
    considered the same ping. This should be a generous upper bound on the
    travel time, probably scaled up by a factor of 2 (the sync between two 
    clocks might be driven by a ping from A->B, and then we look at the error
    of a ping B->A, so the expected error is twice the travel time).


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
    
    amatches=np.asarray(matches)

    # Keep us honest, that the caller didn't try to sneak a bad match in.
    assert np.all( tags_a[amatches[:,0]]==tags_b[amatches[:,1]] )

    # evaluate whether a matched sequence is within tolerance
    a_times=tnums_a[amatches[:,0]]
    b_times=tnums_b[amatches[:,1]]

    if len(a_times)>1:
        mb=np.polyfit(a_times,b_times,1)
        # Be a little smarter about the slope when fitting very short
        # series. 
    else:
        mb=[1.0,b_times[0]-a_times[0]]

    if 0:
        if not np.abs(np.log10(mb[0]))<np.log10(1+max_drift):
            return False # drift is too much
    else:
        # Rather than treating this as an error, instead adjust
        # the fit to the max allowable slope, and below we'll see
        # if the max_error becomes too large
        max_slope=1+max_drift
        min_slope=1./max_slope
        recalc=False
        if mb[0]<min_slope:
            mb[0]=min_slope
            recalc=True
        elif mb[0]>max_slope:
            mb[0]=max_slope
            recalc=True
        if recalc:
            new_b=np.mean(b_times-mb[0]*a_times)
            if verbose:
                print(f"Adjusted slope to be within allowable range, intercept {mb[1]}=>{new_b}")
            mb[1]=new_b

    if verbose:
        print(f"Drift is {1-mb[0]:.4e}")
        print(f"Shift is {np.mean(a_times-b_times):.3f}")
        print("Max error is %.3f"%np.abs(np.polyval(mb,a_times)-b_times).max())
            
    # don't use the intercept -- it's the extrapolated time at tnum epoch.
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

    # This part has to be done per tag
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
        if np.any(~matched):
            # the 0.8 is some slop. 
            n_bad=(deltas[~matched] < 0.8*tagn_interval[tagn]).sum()
            if n_bad>0:
                return False
    return True

def enumerate_matches(tnums_a,tnums_b,tags_a,tags_b,
                      max_shift=20.0,max_drift=0.005,max_delta=0.500,
                      verbose=False):
    """
    Iteratively enumerate allowable sequence matches.
    next_a: index into tnums_a of the next unmatched detection
    next_b: ... tnums_b
    matches: list of index pairs ala next_a/b that have been matched.
    """
    visited={}
    full_matches=[]
    next_a=next_b=0
    matches=[]
    
    # When trying to find a bad ping, keep track of how far the match
    # got. This might be a little generous, in the sense that these
    # are tracked separately, and there could be one set of matches
    # that gets further in a, and a separate set that gets further
    # in b. maybe that's ok, depending on how these values get used.
    max_match_a=-1
    max_match_b=-1
    
    stack=[] # tuples of next_a,next_b,matches
    
    def do_test(next_a,next_b,matches):
        return test_matches(next_a,next_b,matches,tnums_a,tnums_b,tags_a,tags_b,
                            max_shift=max_shift,max_drift=max_drift,max_delta=max_delta)

    stack.append( (next_a,next_b,matches) )

    while stack:
        next_a,next_b,matches = stack.pop()

        # Check to see if this has already been tried
        key=(next_a,next_b,tuple(matches))
        if key in visited: continue
        visited[key]=1

        if verbose:
            print(f"match next_a={next_a}  next_b={next_b}  matches={matches}")

        if not do_test(next_a,next_b,matches):
            continue
        
        if len(matches):
            max_match_a=max(max_match_a,matches[-1][0])
            max_match_b=max(max_match_b,matches[-1][1])
        
        if (next_a==len(tnums_a)) and (next_b==len(tnums_b)):
            # termination condition
            if len(matches)>0:
                full_matches.append(matches)
                print("MATCH")
                continue
        elif (next_a==len(tnums_a)):
            # finish out b
            stack.append( (next_a,len(tnums_b),matches) )
        elif (next_b==len(tnums_b)):
            # finish out a
            stack.append( (len(tnums_a),next_b,matches) )
        else:
            # This ordering will prefer chasing down matches before
            # trying to drop things. Hopefully that means that we'll
            # find the non-empty match first, and could potentially stop
            # early if that is sufficient.
            if tnums_a[next_a]<tnums_b[next_b] + max_shift:
                # drop which ever one is earlier. doesn't work in general, but I'm not
                # explicitly including shifts yet
                stack.append( (next_a+1,next_b,matches) )
            if tnums_b[next_b]<tnums_a[next_a] + max_shift:
                stack.append( (next_a,next_b+1,matches) )

            # is it possible to declare the next two a match?
            if (tags_a[next_a]==tags_b[next_b]):
                time_diff=tnums_a[next_a]-tnums_b[next_b]
                if (np.abs(time_diff)<max_shift):
                    # sure, try a match
                    stack.append( (next_a+1,next_b+1,matches + [(next_a,next_b)]) )
                else:
                    # Tags match, but time diff is too big for max_shift
                    pass

    return full_matches,max_match_a,max_match_b
    

##


# Fit a pair at a time, combining along the way
def merge_detections(ds_a,ds_b,matches):
    amatches=np.asarray(matches)

    tnums_a=ds_a.tnum.values
    tnums_b=ds_b.tnum.values
    times_a=tnums_a[amatches[:,0]]
    times_b=tnums_b[amatches[:,1]]
    tags_a=ds_a.tag.values
    tags_b=ds_b.tag.values
    
    if len(amatches)>1:
        mb=np.polyfit(times_a,times_b,1)
    elif len(amatches)==1:
        mb=[1.0,b_times[0]-a_times[0]]
    else:
        # if at some point we record uncertainty in timestamps, then
        # this would be allowed, and we'd record that pings from one
        # of the sources are basically unconstrained relative to pings
        # from the other sources.  until then, bail.
        raise Exception("Cannot merge when there are no matches")

    tnums_a_adj=np.polyval(mb,tnums_a)

    a_matched=np.zeros(len(tnums_a),np.bool8)
    b_matched=np.zeros(len(tnums_b),np.bool8)
    
    a_matched[amatches[:,0]]=True
    b_matched[amatches[:,1]]=True

    # Unmatched a hits:
    all_tags=np.concatenate( (tags_a[~a_matched],
                              tags_b[~b_matched],
                              tags_a[a_matched]) )
    all_tnums=np.concatenate( (tnums_a_adj[~a_matched],
                               tnums_b[~b_matched],
                               # 0.5*(tnums_a_adj[a_matched]+tnums_b[b_matched])
                               np.minimum(tnums_a_adj[a_matched],tnums_b[b_matched])) )
    a_idxs=np.arange(len(tags_a))
    b_idxs=np.arange(len(tags_b))
    a_srcs=np.concatenate( (a_idxs[~a_matched],
                            -1*np.ones( (~b_matched).sum(), np.int32 ),
                            a_idxs[a_matched]) )
    b_srcs=np.concatenate( (-1*np.ones( (~a_matched).sum(), np.int32 ),
                            b_idxs[~b_matched],
                            b_idxs[b_matched]) )

    
    order=np.argsort(all_tnums)

    all_tnums=all_tnums[order]
    all_tags=all_tags[order]

    ds=xr.Dataset()
    ds['tnum']=('index',), all_tnums
    ds['tag']=('index',), all_tags
    ds['name']=ds_b.name.item()+"-"+ds_a.name.item()

    # Here -- need to record the per-station data.
    # who detected it, at what original time.
    def add_src_matrix(ds):
        if 'matrix' not in ds:
            mat=np.zeros( (ds.dims['index'],1), np.float64)
            mat[:,0]=ds.tnum.values
            ds['rx']=('rx',), [ds.name.item()]
            ds['matrix']=('index','rx'),mat
        return ds
    
    ds_a2=add_src_matrix(ds_a)
    ds_b2=add_src_matrix(ds_b)
    mat_a=ds_a2.matrix.values
    mat_b=ds_b2.matrix.values
    
    detect_matrix=np.zeros( (mat_a.shape[0]+mat_b.shape[0]-len(matches),
                             mat_a.shape[1]+mat_b.shape[1]),
                            np.float64 )
    detect_matrix[:,:]=np.nan # blank slate
    
    # pings from a
    valid=a_srcs[order]>=0
    detect_matrix[valid,:mat_a.shape[1]] = mat_a[a_srcs[order][valid],:]
    valid=b_srcs[order]>=0
    detect_matrix[valid,mat_a.shape[1]:] = mat_b[b_srcs[order][valid],:]
    
    ds['rx']=('rx',), np.concatenate( [ds_a2.rx.values,ds_b2.rx.values] )
    
    ds['matrix']=('index','rx'),detect_matrix
    return ds

def match_sequences(ds_a,ds_b):
    tnums_a=ds_a.tnum.values
    tnums_b=ds_b.tnum.values
    # could be cleaner..
    if 'tagn' in ds_a:
        tags_a=ds_a.tagn.values
    else:
        tags_a=tag_to_num(ds_a.tag.values)
    if 'tagn' in ds_b:
        tags_b=ds_b.tagn.values
    else:
        tags_b=tag_to_num(ds_b.tag.values)
    
    print(f"Aligning {len(tnums_a)} hits at {ds_a.name.item()} with {len(tnums_b)} at {ds_b.name.item()}")
    full_matches,max_a,max_b=enumerate_matches(tnums_a=tnums_a,tnums_b=tnums_b,
                                               tags_a=tags_a,tags_b=tags_b)
    
    print(f"Found {len(full_matches)} potential sets of matches")

    for i,matches in enumerate(full_matches):
        print()
        print(f"---- Match {i}  ({len(matches)} pairs) ---")
        test_matches(len(tnums_a),len(tnums_b),matches,
                     tnums_a=tnums_a,tnums_b=tnums_b,
                     tags_a=tags_a,tags_b=tags_b,
                     verbose=False)

    # For the moment, always pick the first solution
    matches=full_matches[0]

    ds_ab=merge_detections( ds_a,ds_b, matches )
    return ds_ab

ds_ab=match_sequences(clipped_detects[0],
                      clipped_detects[1])
ds_abc=match_sequences(clipped_detects[3],ds_ab)

##

# Plot to see if that is lining up correctly.
# at a glance, seems right
ds=ds_abc

plt.figure(1).clf()

plt.plot( ds.index,
          ds.tnum,
          'o',color='orange',zorder=-1,ms=8,label='matched tnum')

for i,rx in enumerate(ds.rx.values):
    plt.plot( ds.index,
              ds.matrix.isel(rx=i),
              'o',ms=5,label=rx)
plt.legend(loc='upper left')


##


# HERE - roll the bad apple search into match_sequences above.

# Any match between two sequences is allowed to drop 2 detections.
# These are always dropped from sequence 'a'.
# seems that the one bad ping in 2 can take out multiple other pings.
# So incrementally try dropping pings from one side or the other
# until somebody succeeds.
# It's not quite optimal, but at least it gets through the list
# regardless (so far) of order.

max_bad_apples=10

ids=[2,3, 12,
     1, 0, ]
#6, 9, 11,
#     4,5,7,8,10,13 ]

# ids=[0,1,3,12,2,
#      6, 9, 11,
#      4,5,7,8,10,13 ]

# This sequence works.
# ids=[0,1,3,12,4,5,7,8,10,13,9,11,2,6]
#  6 will still fail

# 6(AM8) only detects 45, and nobody else sees 45 (!) FF09

id_b=ids[0]
tnums_b=rx_tnums[id_b]
tags_b=tag_to_num(rx_tags[id_b])
name_b=names[id_b]

for id_a in ids[1:]:
    tnums_a=rx_tnums[id_a]
    tags_a=tag_to_num(rx_tags[id_a])
    name_a=names[id_a]
    
    print(f"Aligning {len(tnums_a)} hits at {name_a} ({id_a}) with {len(tnums_b)} at {name_b}")

    # In some cases a tag is only seen by one or the other set.  no hope for a match,
    # and the current data structure can't record that a subset of the tag detections
    # have an unconstrained clock.  So punt.  A slightly better solution would be
    # to optimize the order of receivers to maximize overlap. Might still arrive
    # at a situation where there is no intersection, or worse, a case where there are
    # two distinct groups, but no commonality between them.
    common_tags=np.intersect1d( np.unique(tags_b),np.unique(tags_a) )
    if len(common_tags)==0:
        print(f"{name_a} and {name_b} have no common tags")
        continue

    bad_apple_search=[ ([], []) ] # each entry a tuple of bad_apples_a, bad_apples_b
    
    while bad_apple_search:
        bad_a,bad_b=bad_apple_search.pop(0) # Breadth-first

        # construct the a and b subsets
        if len(bad_a):
            a_slc=np.ones(len(tnums_a),np.bool8)
            a_slc[ bad_a ]=False
            print(f"Trying bad_apples_a={bad_a}")
        else:        
            a_slc=slice(None)
            
        if len(bad_b):
            b_slc=np.ones(len(tnums_b),np.bool8)
            b_slc[ bad_b ]=False
            print(f"Trying bad_apples_b={bad_b}")
        else:        
            b_slc=slice(None)
            
        full_matches,max_a,max_b=enumerate_matches(tnums_a=tnums_a[a_slc],tnums_b=tnums_b[b_slc],
                                                 tags_a=tags_a[a_slc],tags_b=tags_b[b_slc])
        if len(full_matches):
            break
        else:
            # Currently search by dropping bad pings only on one side at a time
            # if there are no bad apples yet, then queue up a search on both sides.
            # otherwise, queue up a search extending the side we're on.
            if not bad_b: 
                next_bad_a=np.arange(len(tnums_a))[a_slc][max_a+1]
                bad_apple_search.append( (bad_a+[next_bad_a],bad_b))
            if not bad_a:
                next_bad_b=np.arange(len(tnums_b))[b_slc][max_b+1]
                bad_apple_search.append( (bad_a,bad_b+[next_bad_b]) )
    else:
        print(f"Well - the streak has ended. {name_a} and {name_b} yielded no matches")
        break
        
    longest=np.argmax( [len(m) for m in full_matches] )
    print(f"Found {len(full_matches)} potential sets of matches, longest is {len(full_matches[longest])} (i={longest})")

    # update b to be the combination, and here we drop the 'bad apples'
    tnums_b,tags_b=merge_detections(tnums_a[a_slc],tnums_b[b_slc],tags_a[a_slc],tags_b[b_slc],
                                    full_matches[longest])
    name_b=name_b+"-"+name_a
    
##


# They have lots of tags in common.
# print things out see what's up
next_a=next_b=0

count=30
while (next_a<len(tnums_a)) or (next_b<len(tnums_b)):
    if next_a==len(tnums_a):
        sel='b'
    elif next_b==len(tnums_b):
        sel='a'
    elif tnums_a[next_a]<tnums_b[next_b]:
        sel='a'
    else:
        sel='b'
    if sel=='a':
        print(f"[{next_a:4}] {tags_a[next_a]:2} {tnums_a[next_a]:.2f}")
        next_a+=1
    else:
        print(f"          {tnums_b[next_b]:.2f} {tags_b[next_b]:2} [{next_b:4}]")
        next_b+=1
    count-=1
    if count<=0:
        break


    
##

# Did the combine work?
amatches=np.array(full_matches[longest])
apnts=np.array( [tnums_a[amatches[:,0]],
                 0*np.ones(len(amatches))])
bpnts=np.array( [tnums_b[amatches[:,1]],
                 1*np.ones(len(amatches))])

ab_segs=np.array([apnts,bpnts]).transpose(2,0,1)


fig=plt.figure(1)
fig.clf()
ax=fig.add_subplot(1,1,1)

for ii,tnums in enumerate([tnums_a,tnums_b]):
    ax.plot( tnums,ii*np.ones(len(tnums)),
             'o')
ax.add_collection( collections.LineCollection(ab_segs) )

# the combined:
ax.plot( tnums_c,0.5*np.ones(len(tnums_c)),
         'o')

## 
id_d=2 # next up to add in.
tnums_d=rx_tnums[id_d]
tags_d=tag_to_num(rx_tags[id_d])

print(f"Aligning {len(tnums_d)} hits at {names[id_a]} with {len(tnums_c)} from prior")

full_matches=enumerate_matches(tnums_a=tnums_d,tnums_b=tnums_c,
                               tags_a=tags_d,tags_b=tags_c)
longest=np.argmax( [len(m) for m in full_matches] )
print(f"Found {len(full_matches)} potential sets of matches, longest is i={longest} {len(full_matches[longest])}")


