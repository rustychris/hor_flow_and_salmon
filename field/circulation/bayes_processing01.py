"""
Investigate the potential for looking holistically at "all" detections
together, via a Bayesian inference approach.

Slimmed down from bayes_processing.py
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

# tag_rx_count=all_rx_df.groupby(['tag','name']).size().groupby('tag').size().sort_values(ascending=False)
# 
# # 15 tags were seen at more than 1 rx.
# multi_rx_tags= tag_rx_count[ tag_rx_count>1 ].index.values
# 
# grouped=all_rx_df.loc[:,['tag','name','id','tnum']].sort_values(['tag','name']).set_index(['tag','name'])
# 
# for tag,rx_count in tag_rx_count.iteritems():
#     if rx_count<2: break
#     
#     break
# 
# print(f"Tag {tag} was seen by {rx_count} receivers")

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

def test_matches(next_a,next_b,matches,
                 tnums_a,tnums_b,tags_a,tags_b,
                 verbose=False,
                 max_shift=20.0,max_drift=0.005,max_delta=0.500,
                 max_bad_pings=0):
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

    # This part has to be done per tag:
    n_bad=0 # running tally of number of bad pings
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
            n_bad+=(deltas[~matched] < 0.8*tagn_interval[tagn]).sum()
            if n_bad>max_bad_pings:
                # the 0.8 is some slop. 
                return False
            elif (n_bad>0) and verbose:
                print("%d bad pings, but that is allowed"%n_bad)
    return True

# for the purpose of matching just two sequences, shift
# and drift are two numbers.

def match_remainder(next_a,next_b,matches,
                    tnums_a,tnums_b,tags_a,tags_b,
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

id_a=0
id_b=1

tnums_a=rx_tnums[id_a]
tnums_b=rx_tnums[id_b]

tags_a=tag_to_num(rx_tags[id_a])
tags_b=tag_to_num(rx_tags[id_b])
    
print(f"Aligning {len(tnums_a)} hits at {names[id_a]} with {len(tnums_b)} at {names[id_b]}")
full_matches,max_a,max_b=match_remainder(0,0,[],
                                         tnums_a=tnums_a,tnums_b=tnums_b,
                                         tags_a=tags_a,tags_b=tags_b)

print(f"Found {len(full_matches)} potential sets of matches")

for i,matches in enumerate(full_matches):
    print()
    print(f"---- Match {i}  ({len(matches)} pairs) ---")
    test_matches(len(tnums_a),len(tnums_b),matches,
                 tnums_a=tnums_a,tnums_b=tnums_b,
                 tags_a=tags_a,tags_b=tags_b,
                 verbose=False)

##

# First go at more than 2 receivers.
# Fit a pair at a time, combining along the way
def merge_detections(tnums_a,tnums_b,tags_a,tags_b,matches):
    amatches=np.asarray(matches)
    times_a=tnums_a[amatches[:,0]]
    times_b=tnums_b[amatches[:,1]]
    if len(amatches)>1:
        mb=np.polyfit(times_a,times_b,1)
    elif len(amatches)==1:
        mb=[1.0,b_times[0]-a_times[0]]
    else:
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
    
    order=np.argsort(all_tnums)

    all_tnums=all_tnums[order]
    all_tags=all_tags[order]
    
    return all_tnums,all_tags

##

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
            
        full_matches,max_a,max_b=match_remainder(0,0,[],
                                                 tnums_a=tnums_a[a_slc],tnums_b=tnums_b[b_slc],
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

full_matches=match_remainder(0,0,[],
                             tnums_a=tnums_d,tnums_b=tnums_c,
                             tags_a=tags_d,tags_b=tags_c)
longest=np.argmax( [len(m) for m in full_matches] )
print(f"Found {len(full_matches)} potential sets of matches, longest is i={longest} {len(full_matches[longest])}")





##


          4492794.04 43 [   0]
          4492802.23 47 [   1]
          4492805.57 38 [   2]
          4492806.39 30 [   3]
[   0] 39 4492811.58
[   1] 44 4492816.49
          4492818.76 39 [   4]
          4492821.36 26 [   5]
          4492822.54 40 [   6]
          4492823.57 44 [   7]
          4492823.98 42 [   8]
          4492824.03 49 [   9]
          4492824.80 43 [  10]
          4492826.73 41 [  11]
          4492828.70 48 [  12]
          4492831.63 50 [  13]
          4492831.96 52 [  14]
          4492832.66 37 [  15]
          4492833.74 31 [  16]
          4492834.04 47 [  17]
          4492835.85 38 [  18]
          4492836.04  9 [  19]
          4492838.80 31 [  20]
          4492841.64 26 [  21]
[   2] 39 4492841.96
          4492848.92 31 [  22]
          4492849.14 39 [  23]
          4492853.02 40 [  24]
          4492854.45 44 [  25]
          4492854.66 42 [  26]
          4492855.58 43 [  27]
          4492856.41 49 [  28]
[   3] 50 4492857.22
          4492857.31 41 [  29]
[   4] 38 4492859.03
          4492859.04 31 [  30]
