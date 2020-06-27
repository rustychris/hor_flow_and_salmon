"""
Pull out just the ping matching code

"""

import os
import pandas as pd
import numpy as np
import xarray as xr
import six
from stompy import utils

import parse_tek as pt

##

def rx_locations_2019():
    # Need utm for each of the 12 receivers.
    # 'SM8', 'SM7', 'SM4', 'SM3', 'SM2', 'AM9', 'AM7', 'AM6', 'AM4',
    # 'AM3', 'AM2', 'AM1'

    deploy2019=pd.read_csv('2019 deployment.csv',names=['shot','y','x','z'],index_col=False)
    survout=pd.read_csv('multid_survout.csv',names=['shot','y','x','z'],index_col=False)
    shots=pd.concat([deploy2019,survout],axis=0)
    shots['rx']=[ rx.split('.')[0].upper() for rx in shots.shot.values]

    grped=shots.groupby('rx').mean()
    return grped

def rx_locations_2018():
    shots=pd.read_csv("2018_Data/monitor deployment_swap-sm2-sm3.csv",
                      names=['shot','y','x','z'],index_col=False)
    shots['rx']=[ rx.split('.')[0].upper() for rx in shots.shot.values]
    grped=shots.groupby('rx').mean()
    return grped

## 
if 0: # 2019 data:
    rx_locations=rx_locations_2019()
    all_receiver_fns=[
        ('AM1',"2019 Data/HOR_Flow_TekDL_2019/AM1_186002/am18-6002190531229.DET"),
        ('AM2',"2019 Data/HOR_Flow_TekDL_2019/AM2_186006/20190517/186006_190222_130811_P/186006_190222_130811_P.DET"),
        ('AM3',"2019 Data/HOR_Flow_TekDL_2019/AM3_186003/20190517/186003_190217_125944_P/186003_190217_125944_P.DET"),
        ('AM4',"2019 Data/HOR_Flow_TekDL_2019/AM4_186008/20190517/186008_190517_130946_P/186008_190517_130946_P.DET"),
        ('AM6',"2019 Data/HOR_Flow_TekDL_2019/AM6_186005/20190517/186005_190224_141058_P/am18-6005190481424.DET"),
        ('AM7',"2019 Data/HOR_Flow_TekDL_2019/AM7_186007/20190517/186007_190217_135133_P/am18-6007190481351.DET"),
        ('AM8',"2019 Data/HOR_Flow_TekDL_2019/AM8_186009/_186009/186009_190222_132321_F/186009_190222_132321_F.DET"),
        ('AM9',"2019 Data/HOR_Flow_TekDL_2019/AM9_186004/20190517/186004_190311_144244_P/am18-6004190440924.DET"),
        ('SM1',"2019 Data/HOR_Flow_TekDL_2019/SM1_187023/20190517/187023_190416_050506_P/sm18-7023190711049.DET"),
        ('SM2',"2019 Data/HOR_Flow_TekDL_2019/SM2_187013/2018_7013/187013_190516_143749_P.DET"),
        ('SM3',"2019 Data/HOR_Flow_TekDL_2019/SM3_187026/20190516/187026_190301_124447_P/sm18-7026190421300.DET"),
        ('SM4',"2019 Data/HOR_Flow_TekDL_2019/SM4_187020/20190516/187020_190301_123034_P/sm18-7020190421306.DET"),
        ('SM7',"2019 Data/HOR_Flow_TekDL_2019/SM7_187017/20190516/187017_190225_154939_P/sm18-7017190391457.DET"),
        ('SM8',"2019 Data/HOR_Flow_TekDL_2019/SM8_187028/20190516/187028_190216_175239_P/sm18-7028190421324.DET")
    ]


## 

def test_matches_ds(next_a,next_b,matches,ds_a,ds_b,**kw):
    return test_matches(next_a,next_b,matches,
                        ds_a.tnum.values,ds_b.tnum.values,
                        tag_to_num(ds_a.tag.values),
                        tag_to_num(ds_b.tag.values),
                        **kw)

# Core of the matching algorithm:
#   test_matches(), match_remainder()
#   These should be kept to simple (python and numpy types)
#   inputs, and left as standalone methods.  At some point
#   they may be optimized via numba and that will be easier
#   with simple inputs.
def test_matches(next_a,next_b,matches,
                 tnums_a,tnums_b,tags_a,tags_b,
                 tagn_interval,
                 verbose=False,
                 max_shift=20.0,max_drift=0.005,max_delta=0.500):
    """
    return True if the matches so far, having considered
    up to next_a and next_b, are possible.
    In the case that constraints are violated, if specific culprits
    can be identified return them as a tuple ( [bad a samples], [bad b samples] )
    otherwise, return False.


    Search state:
    matches: a list of tuples [ (a_idx,b_idx), ... ]
      noting which pings are potentially the same.

    next_a,next_b: index into tnums_a, tnums_b for the next detection
     not yet considered.
    
    tnums_a,tnums_b: timestamps for the pings.
    tags_a,tags_b:  tag ids (numbers, not strings) for the pings.

    tagn_interval: a dict mapping tagn to expected interval between pings in 
     seconds.

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
    if len(matches)==0:
        # is a too far ahead of b?
        if next_a>0 and next_b<len(tnums_b) and tnums_a[next_a-1]>tnums_b[next_b] + max_shift:
            # we already looked at an a that comes after the next b.
            return ([next_a-1],[next_b])
        if next_b>0 and next_a<len(tnums_a) and tnums_b[next_b-1]>tnums_a[next_a] + max_shift:
            # already looked at a b that comes after the next a
            return ([next_a],[next_b-1])
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

    # Rather than treating too large a drift as an error, instead adjust
    # the fit to the max allowable slope, and below we'll see
    # if the max_error becomes too large.  Otherwise fits to short strings of
    # data may misconstrue noise as drift and get an erroneously large drift.
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

    abs_errors=np.abs(np.polyval(mb,a_times)-b_times)
    if abs_errors.max() >= max_delta:
        bad_match=np.argmax(abs_errors)
        bad_pair=( [matches[bad_match][0]],
                   [matches[bad_match][1]] )
        # print("Abs error too great - bad_pair is ",bad_pair)
        return bad_pair # resulting travel times are too much.

    # to consider: instead of just going through next_a and next_b,
    #  this could go through the later of the adjusted times
    dtype=[('tnum',np.float64),
           ('matched',np.bool8),
           ('tag',np.int32),
           ('src','S1'),
           ('srci',np.int32)]
    a_full=np.zeros(next_a,dtype=dtype)
    b_full=np.zeros(next_b,dtype=dtype)
    
    a_full['tnum']=np.polyval(mb,tnums_a[:next_a])
    b_full['tnum']=tnums_b[:next_b]
    a_full['matched'][amatches[:,0]]=True
    b_full['matched'][amatches[:,1]]=True
    a_full['tag']=tags_a[:next_a]
    b_full['tag']=tags_b[:next_b]
    a_full['src']='a'
    b_full['src']='b'
    a_full['srci']=np.arange(next_a)
    b_full['srci']=np.arange(next_b)

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
            bad=(~matched) & (deltas < 0.8*tagn_interval[tagn])
            bad_a=[]
            bad_b=[]
            
            for bad_idx in np.nonzero(bad)[0]:
                # deltas[bad_idx] is bad
                # That's from idxs[bad_idx],idxs[bad_idx+1]
                for bad_i in [bad_idx,bad_idx+1]:
                    i=idxs[bad_i]
                    if combined['src'][i]==b'a':
                        bad_a.append( combined['srci'][i] )
                    elif combined['src'][i]==b'b':
                        bad_b.append( combined['srci'][i] )
                    else:
                        assert False
                        
            if bad.sum()>0:
                bad_a=list(np.unique(bad_a))
                bad_b=list(np.unique(bad_b))
                # print("Per tag delta is bad.  Pair is ",(bad_a,bad_b))
                return (bad_a,bad_b)
    return True

## 
def enumerate_matches(tnums_a,tnums_b,tags_a,tags_b,
                      tagn_interval,
                      max_shift=20.0,max_drift=0.005,max_delta=0.500,
                      verbose=False,max_matches=1):
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
    # 2019-12-03: HERE so far this has tracked the last matched ping, and
    # on failure started axing successive pings.
    # This is problematic because we may consider many ping after the
    # last match, and much later consider a pair that fails.
    # tempting to track the greatest next_a/next_b while building the
    # longest match. but the search can go quite far (up to max shift)
    # on either side without having to advance the other side, such
    # that we may see very far beyond the real culprit.
    # Solution probably to make test smarter, and allow it to return
    # potential bad apples.
    max_match_a=-1
    max_match_b=-1

    # New approach to finding bad apples.
    # Keep the test result from the longest set of matches
    longest_fail=None
    max_matched=0
    longest_match=None
    
    stack=[] # tuples of next_a,next_b,matches
    
    def do_test(next_a,next_b,matches):
        return test_matches(next_a,next_b,matches,tnums_a,tnums_b,tags_a,tags_b,
                            tagn_interval=tagn_interval,
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

        test_result=do_test(next_a,next_b,matches)
        if test_result is not True:
            if len(matches)>max_matched:
                max_matched=len(matches)
                longest_fail=test_result
                longest_match=matches
            elif len(matches)==max_matched:
                longest_fail=( list(set( longest_fail[0]+test_result[0])), 
                               list(set( longest_fail[1]+test_result[1])) )
            continue
        
        if len(matches):
            max_match_a=max(max_match_a,matches[-1][0])
            max_match_b=max(max_match_b,matches[-1][1])
        
        if (next_a==len(tnums_a)) and (next_b==len(tnums_b)):
            # termination condition
            if len(matches)>0:
                full_matches.append(matches)
                print("MATCH")
                if len(full_matches)>=max_matches:
                    break
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

    return full_matches,max_match_a,max_match_b,longest_fail



def print_sequence_pair(ds_a,ds_b,count=30,tag_sel=None):
    # They have lots of tags in common.
    # print things out see what's up
    next_a=next_b=0

    tnums_a=ds_a.tnum.values
    tnums_b=ds_b.tnum.values
    tags_a=ds_a.tag.values
    tags_b=ds_b.tag.values

    if tag_sel=='common':
        tag_sel=np.intersect1d( np.unique(tags_a),
                                np.unique(tags_b) )
        print("Common tags: ",tag_sel)
        
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
            if (tag_sel is None) or (tags_a[next_a] in tag_sel):
                print(f"[{next_a:4}] {tags_a[next_a]:4} {tnums_a[next_a]:.2f}")
            next_a+=1
        else:
            if (tag_sel is None) or (tags_b[next_b] in tag_sel):
                print(f"            {tnums_b[next_b]:.2f} {tags_b[next_b]:2} [{next_b:4}]")
            next_b+=1
        count-=1
        if count<=0:
            break


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
        mb=[1.0,times_b[0]-times_a[0]]
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
            # also any other fields known to be rx-specific
            for field in ['rx_x','rx_y','rx_z']:
                if field in ds:
                    ds[field]=('rx',), [ds[field].values]
            # and sensor data related to a ping at a rx
            for field in ['temp','pressure','corrQ','nbwQ']:
                vals=ds[field].values[:,None]
                ds[field]=('index','rx'),vals
        return ds

    valid_a=a_srcs[order]>=0
    valid_b=b_srcs[order]>=0
    
    ds_a2=add_src_matrix(ds_a)
    ds_b2=add_src_matrix(ds_b)

    for mat_field in ['matrix','temp','pressure','corrQ','nbwQ']:
        mat_a=ds_a2[mat_field].values
        mat_b=ds_b2[mat_field].values
    
        mat_ab=np.zeros( (mat_a.shape[0]+mat_b.shape[0]-len(matches),
                          mat_a.shape[1]+mat_b.shape[1]),
                         mat_a.dtype )
        if np.issubdtype(mat_ab.dtype,np.floating):
            missing=np.nan
        else:
            missing=-1
            
        mat_ab[:,:]=missing # blank slate
        # pings from a
        mat_ab[valid_a,:mat_a.shape[1]] = mat_a[a_srcs[order][valid_a],:]
        # pings from b
        mat_ab[valid_b,mat_a.shape[1]:] = mat_b[b_srcs[order][valid_b],:]
        ds[mat_field]=('index','rx'),mat_ab

    for field in ds_a2.variables: 
        if ds_a2[field].dims!=('rx',):
            continue
        if field not in ds_b2:
            continue
        ds[field]=('rx',),np.concatenate( [ds_a2[field].values,
                                           ds_b2[field].values] )
    
    return ds

        
class PingMatcher(object):
    T0=None # set to a datetime64 around the time of the field data
    max_shift=20 # seconds
    max_drift=0.005 # seconds/second
    max_delta=0.500 # seconds
    max_bad_pings=10
    verbose=False
    max_matches=1 # stop when this many potential sets of matches are found

    clipped=None

    # Originally, when searching for bad pings, the assumption was that only
    # one dataset or the other had bad pings.  The relatively low rate of
    # bad pings allowed this.  But in processing all of the 2018 data,
    # it appears that at least on occasion, during a 6 hour window, both datasets
    # can have some bad pings
    allow_dual_bad_pings=False
    
    def __init__(self,**kw):
        utils.set_keywords(self,kw)
        self.all_detects=[]
        
    def add_detections(self,name,det_fn,**kw):
        detects=pt.parse_tek(det_fn,name=name,**kw)
        if isinstance(detects,list):
            for i,d in enumerate(detects):
                d['station']=d['name']
                d['name']=(),d['name'].item()+'.%d'%i
                self.all_detects.append(d)
        else:
            detects['station']=detects['name']
            self.all_detects.append(detects)
    def copy(self):
        pm=PingMatcher()
        pm.all_detects=self.all_detects
        pm.T0=self.T0
        pm.allow_dual_bad_pings=self.allow_dual_bad_pings
        return pm
    
    def remove_multipath(self):
        pm=self.copy()
        pm.all_detects=[pt.remove_multipath(d) for d in self.all_detects]
        return pm
    def clip_time(self,t_clip):
        pm=self.copy()
        pm.all_detects=[ d.isel( index=(d.time.values>=t_clip[0]) & (d.time.values<t_clip[-1]))
                         for d in self.all_detects ]
        pm.clipped=t_clip
        return pm
    
    def to_tnum(self,x):
        """
        Local mapping of datetime64 to application specific time value.
        Currently floating point seconds since 2019-02-01, but trying
        to abstract the details away
        """
        return (x-self.T0)/np.timedelta64(1,'ns') * 1e-9

    def from_tnum(self,x):
        return self.T0 + x*1e9*np.timedelta64(1,'ns')

    def tag_to_num(self,t):
        # for enumerate_groups tag must be numeric
        # Tags are hex, so just parse the hex
        if isinstance(t,(str,bytes)):
            return int(t,16)
        else:
            return np.array( [self.tag_to_num(x) for x in t] )
    
    def match_sequences(self,ds_a,ds_b):
        tnums_a=ds_a.tnum.values
        tnums_b=ds_b.tnum.values
        # could be cleaner..
        if 'tagn' in ds_a:
            tags_a=ds_a.tagn.values
        else:
            tags_a=self.tag_to_num(ds_a.tag.values)
        if 'tagn' in ds_b:
            tags_b=ds_b.tagn.values
        else:
            tags_b=self.tag_to_num(ds_b.tag.values)

        print(f"Aligning {len(tnums_a)} hits at {ds_a.name.item()} with {len(tnums_b)} at {ds_b.name.item()}")

        bad_apple_search=[ ([], []) ] # each entry a tuple of bad_apples_a, bad_apples_b

        while bad_apple_search:
            bad_a,bad_b=bad_apple_search.pop(0) # Breadth-first
            if len(bad_a)+len(bad_b)>self.max_bad_pings:
                continue # too long

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

            full_matches,max_a,max_b,longest_fail=enumerate_matches(tnums_a=tnums_a[a_slc],tnums_b=tnums_b[b_slc],
                                                                    tags_a=tags_a[a_slc],tags_b=tags_b[b_slc],
                                                                    tagn_interval=self.tagn_interval,
                                                                    max_shift=self.max_shift,max_drift=self.max_drift,max_delta=self.max_delta,
                                                                    verbose=self.verbose,max_matches=self.max_matches)

            if len(full_matches):
                if len(bad_a):
                    ds_a=ds_a.isel(index=a_slc)
                if len(bad_b):
                    ds_b=ds_b.isel(index=b_slc)
                break
            else:
                # for mapping back to original indices
                a_slc_idx=np.arange(len(tnums_a))[a_slc]
                b_slc_idx=np.arange(len(tnums_b))[b_slc]

                if longest_fail is not None:
                    print("Longest fail: ", longest_fail)
                    if (not bad_b) or (self.allow_dual_bad_pings):
                        for a in longest_fail[0]:
                            next_bad_a=a_slc_idx[a]
                            bad_apple_search.append( (bad_a+[next_bad_a], bad_b) )
                    if (not bad_a) or (self.allow_dual_bad_pings):
                        for b in longest_fail[1]:
                            next_bad_b=b_slc_idx[b]
                            bad_apple_search.append( (bad_a, bad_b+[next_bad_b]) )
                else:
                    # hoping that this code will become obsolete
                    print("Fallback to old bad apple code")
                    # Currently search by dropping bad pings only on one side at a time
                    # if there are no bad apples yet, then queue up a search on both sides.
                    # otherwise, queue up a search extending the side we're on.
                    if not bad_b:
                        next_bad_a=a_slc_idx[max_a+1]
                        bad_apple_search.append( (bad_a+[next_bad_a],bad_b))
                    if not bad_a:
                        next_bad_b=b_slc_idx[max_b+1]
                        bad_apple_search.append( (bad_a,bad_b+[next_bad_b]) )
        else:
            raise Exception("Failed to find matches!")

        longest=np.argmax( [len(m) for m in full_matches] )
        print(f"Found {len(full_matches)} potential sets of matches, longest is {len(full_matches[longest])} (i={longest})")

        # At some point it may be necessary to further investigate multiple sets of matches.
        # for now, assume the longest is the best and forge ahead.
        matches=full_matches[longest]

        ds_ab=merge_detections( ds_a,ds_b, matches )
        return ds_ab

    def prepare(self):
        # scan to get complete list of receivers and tags
        # maybe not used anymore.
        #self.all_rx_names=[ ds.name.item() for ds in self.all_detects]
        self.all_tags=np.unique( np.concatenate( [np.unique(ds.tag.values) for ds in self.all_detects] ) )

        self.beacon_tagns=self.tag_to_num( [ds.beacon_id.item() for ds in self.all_detects] )

        tagn_interval={}
        for t in self.all_tags:
            tagn=self.tag_to_num(t)
            if tagn in self.beacon_tagns:
                # beacon tags every 30 seconds
                tagn_interval[tagn]=30.0
            else:
                # fish tags every 5 seconds
                tagn_interval[tagn]=5.0
        self.tagn_interval=tagn_interval

        for d in self.all_detects:
            d['tnum']=('index',),self.to_tnum(d['time'].values)

    def match_all(self):
        """
        detects: list of xr.Dataset, each one giving the time and tags from a single
        receiver.

        returns a merged dataset matching up as many of the pings as possible.
        """
        self.prepare()
        detects=self.all_detects

        idxs=range(len(self.all_detects))

        ds_b=detects[0]

        for idx_a in idxs[1:]:
            print(f"Adding idx {idx_a} into the mix")
            ds_a=detects[idx_a]
            if len(ds_a.index)==0: continue # probably nixed during clip_time()
            tags_b=self.tag_to_num(ds_b.tag.values)
            tags_a=self.tag_to_num(ds_a.tag.values)

            # In some cases a tag is only seen by one or the other set.  no hope for a match,
            # and the current data structure can't record that a subset of the tag detections
            # have an unconstrained clock.  So punt.  A slightly better solution would be
            # to optimize the order of receivers to maximize overlap. Might still arrive
            # at a situation where there is no intersection, or worse, a case where there are
            # two distinct groups, but no commonality between them.
            common_tags=np.intersect1d( np.unique(tags_b),np.unique(tags_a) )

            if len(common_tags)==0:
                print(f"{ds_a.name.item()} and {ds_b.name.item()} have no common tags")
                continue

            ds_b=self.match_sequences(ds_a,ds_b)

        self.add_beacon_list(ds_b)    

        return ds_b

    def match_all_by_similarity(self):
        """
        Similar to match_all(), but process the groups in order of best 
        potential match. In some cases this is more successful than match_all().
        match_all() uses the given order of receivers. 
        For example, with receivers [A,B,C], it may be the case that 
        A-B is poorly constrained, but A-C and B-C are constrained.  match_all()
        may get A-B wrong, which then sinks the rest of the matching.
        The approach here evaluates each pair in terms of the maximum possible 
        number of matches, and matches the pair with the greatest possible
        number of matches first.  This is repeated iteratively until no
        more potential matches are possible.

        detects: list of xr.Dataset, each one giving the time and tags from a single
        receiver.

        returns a merged dataset matching up as many of the pings as possible, or None
          in the case that there are no possible beacon tag alignments. That will
          happen when no receiver hear another receiver.  Note that this ignores
          non-beacon tags that might be heard by multiple receivers and allow a coarse
          clock alignment (ignored because most likely there won't enough data to 
          complete trilateration)
        """
        self.prepare()
        beacon_tags=np.unique([ds.beacon_id.item() for ds in self.all_detects])

        # Filter down to rx that actually have some pings
        dss=[ds for ds in self.all_detects if ds.dims['index']]

        while len(dss)>1:
            Ndet=len(dss)

            tag_counts=[ [ (ds.tag.values==b).sum() for b in beacon_tags ]
                         for ds in dss ]

            best_match=[0,None]
            for ds_ai in range(Ndet):
                for ds_bi in range(ds_ai+1,Ndet):
                    # dss can contain sequential instances of the same stations,
                    # before/after a clock reset.  Would like to ignore cases where station
                    # is the same.  Not that simple.  Only the original datasets
                    # have station.
                    # If it's not ignored here, then we risk match_sequences failing
                    # below, and then we won't know how to proceed.
                    # Instead, ignore cases where the times do not significantly overlap.
                    # Still not great...
                    common_start=max(dss[ds_ai].tnum.values[0], dss[ds_bi].tnum.values[0])
                    common_end = min(dss[ds_ai].tnum.values[-1],dss[ds_bi].tnum.values[-1])
                    t_overlap=common_end-common_start
                    # don't process if less than 10 minutes. Note that big clock shifts
                    # like an hour will screw this up, but none of this process can deal
                    # with errors that large.
                    if t_overlap<600.0: 
                        continue
                    
                    a_tag_counts=tag_counts[ds_ai]
                    b_tag_counts=tag_counts[ds_bi]
                    # calculate a max possible number of matches
                    ab_tag_counts=[ min(a,b) for a,b in zip(a_tag_counts,b_tag_counts)]
                    sim=np.sum(ab_tag_counts)
                    if sim>best_match[0]:
                        best_match=[sim,(ds_ai,ds_bi)]
            if best_match[0]==0:
                break
            ai,bi=best_match[1]
            print(f"Will match ai={ai} bi={bi} with best possible length of {best_match[0]}")
            ab=self.match_sequences(dss[ai], dss[bi])
            dss=dss[:ai] + dss[ai+1:bi] + dss[bi+1:]
            dss.append(ab)

        # Scan the remaining datasets, and hope that exactly one has merged results.
        merged_dss=[ds for ds in dss if ds.dims.get('rx',0)>0 ]
        if len(merged_dss)==0:
            return None
        elif len(merged_dss)>1:
            # Probably an indication of a bug.  It *could* be real, in which case
            # the easiest solution is to return the one with the most total pings,
            # but that would discard the information in the other connected components.
            raise Exception("Multiple connected components of rx.  Not ready for that")
        else:
            ds_result=merged_dss[0]
            self.add_beacon_list(ds_result)    
            return ds_result
    
    def set_rx_locations(self,df):
        """
        Save UTM coordinate info.  df should be indexed
        by upper-case receiver name, and have x and y columns.
        """
        self.rx_locs=df
        for d in self.all_detects:
            name=d.station.item()
            if name in self.rx_locs.index:
                x,y,z=self.rx_locs.loc[name,['x','y','z']].values
            else:
                x=y=z=np.nan
            d['rx_x']=x
            d['rx_y']=y
            d['rx_z']=z

    def add_beacon_list(self,ds_total):
        """
        annotate each ping with the beacon ID it came from if
        it was from a beacon.  mostly just to mark that it's a beacon
        ping
        """
        beacons=[]

        for rx in ds_total.rx.values:
            for ds in self.all_detects:
                if ds.name==rx:
                    beacons.append(ds.beacon_id.item())
                    break
            else:
                beacons.append(None)
        ds_total['rx_beacon']=('rx',),beacons
        return ds_total


def ping_matcher_2018(**kw):
    """
    Load 2018 data.  This is intended as the single starting point 
    for processing 2018 data
    
    keywords arguments passed to add_detections()
    """
    pm=PingMatcher()
    pm.T0=np.datetime64('2018-02-01 00:00')

    # 2018 data. no pressure data.
    # Also no CF2 files?  are there any files that do give a beacon id?
    # in DBG, NoiseAlpha=0xFFBE. probably not useful.
    # AM2 should be FF08, based on hearing itself.

    pm.add_detections(name='AM1',det_fn='2018_Data/AM1_20186009/050318/AM1180711347.DET',
                      pressure_range=None,**kw)
    pm.add_detections(name='AM2',det_fn='2018_Data/AM2_20186008/043018/AM2180711401.DET',
                      pressure_range=None,**kw)
    pm.add_detections(name='AM3',det_fn='2018_Data/AM3_20186006/043018/2018-6006180621239.DET',
                      pressure_range=None,**kw)
    pm.add_detections(name='AM4',det_fn='2018_Data/AM4_20186005/050118/2018-6005180671524.DET',
                      pressure_range=None,**kw)
    pm.add_detections(name='AM5',det_fn='2018_Data/AM5_20186004/050318/2018-6004180671635.DET',
                      pressure_range=None,**kw)
    pm.add_detections(name='AM8',det_fn='2018_Data/AM8_20186002/043018/2018-6002180401450.DET',
                      pressure_range=None,**kw)
    pm.add_detections(name='AM9',det_fn='2018_Data/AM9_20186001/043018/AM9180711355.DET',
                      pressure_range=None,**kw)
    pm.add_detections(name='SM1',det_fn='2018_Data/SM1_20187013/050218/SM1180711252.DET',
                      pressure_range=None,**kw)
    pm.add_detections(name='SM2',det_fn='2018_Data/SM2_20187014/050218/SM2180711257.DET',
                      pressure_range=None,**kw)
    pm.add_detections(name='SM3',det_fn='2018_Data/SM3_20187015/SM3180711301.DET',
                      pressure_range=None,**kw)
    pm.add_detections(name='SM4',det_fn='2018_Data/SM4_20187018/050218/SM6180711928.DET',
                      pressure_range=None,**kw)
    pm.add_detections(name='SM8',det_fn='2018_Data/SM8_20187017/050218/SM5180711921.DET',
                      pressure_range=None,**kw)
    pm.add_detections(name='SM9',det_fn='2018_Data/SM9_20187024/050218/SM12180721837.DET',
                      pressure_range=None,**kw)

    pm.set_rx_locations(rx_locations_2018())

    return pm

if 0:
    # see notes in bayes_v02 on picking up clock resets

    pm_nomp=pm.remove_multipath()

    pm_clip=pm_nomp.clip_time([np.datetime64("2018-03-20 20:00"),
                               np.datetime64("2018-03-20 22:00")])

    ds_total=pm_clip.match_all()

    ##

    fn=f'pings-{str(pm_clip.clipped[0])}_{str(pm_clip.clipped[1])}.nc'
    os.path.exists(fn) and os.unlink(fn)
    ds_total.to_netcdf(fn)

