import pandas as pd
import re
import logging as log
import xarray as xr
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
from stompy import utils


def parse_tek(det_fn,cf2_fn=None,name=None,pressure_range=[110e3,225e3],
              auto_beacon=True,split_on_clock_change=True):
    """
    det_fn: path to DET file with detection information
    cf2_fn: optional, read beacon id from CF2 file.  Will attempt to
      guess this if not specified. 
    name: string identifier added to dataset
    pressure_range: valid range of pressures, used to filter the time series
     to when the receiver was in the water.  pass None to skip any filtering.
    auto_beacon: if beacon tag ID cannot be read from CF2, this enables choosing
     the most common received tag as the beacon id.
    split_on_clock_change: if true, return a list of datasets, split up based on
      when logs indicated that the clock was updated.
    """
    if cf2_fn is None:
        fn=det_fn.replace('.DET','.CF2')
        if os.path.exists(fn):
            cf2_fn=fn

    df=pd.read_csv(det_fn,
                   names=['id','year','month','day','hour','minute','second','epoch','usec',
                          'tag','nbwQ','corrQ','num1','num2','one','pressure','temp'])

    # this is quite fast and should yield UTC.
    # the conversion in utils happens to be good down to microseconds, so we can
    # do this in one go
    df['time'] = utils.unix_to_dt64(df['epoch'] + df['usec']*1e-6)
        
    # clean up time:
    bad_time= (df.time<np.datetime64('2018-01-01'))|(df.time>np.datetime64('2022-01-01'))
    df2=df[~bad_time].copy()

    # clean up temperature:
    df2.loc[ df.temp<-5, 'temp'] =np.nan
    df2.loc[ df.temp>35, 'temp'] =np.nan

    # clean up pressure
    # this had been limited to 160e3, but
    # AM9 has a bad calibration (or maybe it's just really deep)
    if pressure_range is not None:
        df2.loc[ df2.pressure<pressure_range[0], 'pressure']=np.nan
        df2.loc[ df2.pressure>pressure_range[1], 'pressure']=np.nan

    # trim to first/last valid pressure
    valid_idx=np.nonzero( np.isfinite(df2.pressure.values) )[0]
    df3=df2.iloc[ valid_idx[0]:valid_idx[-1]+1, : ].copy()

    df3['tag']=[ s.strip() for s in df3.tag.values ]

    ds=xr.Dataset.from_dataframe(df3)
    ds['det_filename']=(),det_fn

    if name is not None:
        ds['name']=(),name

    if cf2_fn is not None:
        # SM2 isn't getting the right value here.
        # looks like it would be FF13, but it never
        # hears FF13.
        cf=pd.read_csv(cf2_fn,header=None)
        local_tag=cf.iloc[0,1].strip()
        ds['beacon_id']=local_tag
        ds['cf2_filename']=(),cf2_fn
    elif auto_beacon:
        beacon_id=df3.groupby('tag').size().sort_values().index[-1]
        ds['beacon_id']=beacon_id
        ds['cf2_filename']=None
        ds['beacon_id'].attrs['source']='received tags'

    ds.attrs['pressure_range']=pressure_range
    if split_on_clock_change:
        dbg_filename=ds.det_filename.item().replace('.DET','.DBG')
        if not os.path.exists(dbg_filename):
            print("Split on clock: couldn't find %s"%dbg_filename)
            return [ds]
        else:        
            all_clock_resets=dbg_to_clock_changes(dbg_filename)
            diced=dice_by_clock_resets(ds,all_clock_resets)
            return diced
    else:
        return ds

def remove_multipath(ds):
    # Sometimes a single ping is heard twice (a local bounce)
    # when the same tag is heard in quick succession (<1s) drop the
    # second occurrence.
    # ds: Dataset filtered to a single receiver and a single beacon id.
    delta_us=np.diff(ds.time.values) / np.timedelta64(1,'us')

    if np.any(delta_us<=0):
        log.warning("%s has non-increasing timestamps (worst is %.2fs)"%(ds.det_filename.values,
                                                                         delta_us.min()*1e-6))
    # Warn and drop any non-monotonic detections.
    bounces=(delta_us<1e6) & (delta_us>0)
    valid=np.r_[ True, delta_us>1e6 ]
    return ds.isel(index=valid).copy()

def dbg_to_clock_changes(dbg_filename):
    """
    Scan the .DBG file for indications that the FPGA clock was changed.
    Returns a DataFrame with fpga_pre and fpga_post fields giving UTC time
    before and just after clock was changed. For unanticipated resets, the 
    DBG file just has hourly updates, so fpga_pre could be as much as an hour
    before the clock was actually reset.
    """
    dbg=pd.read_csv(dbg_filename,names=['time','message'])
    idx=0
    clock_resets=[] # (time_before,msg_before,time_after, msg_after)

    # For reboots, track the last time we got a DBG message about the FPGA
    # clock
    last_fpga_status=dict(time_pre=None,msg_pre=None,fpga_pre=None)

    def parse_d(s):
        return datetime.datetime.strptime(s,'%m/%d/%Y %H:%M:%S')

    while idx<len(dbg):
        # Seem to get these status messages hourly
        # 03/24/2019 15:25:31, RTC: 1553469930.648437 FPGA: 1553469931.000051 dt=-0.351614
        m=re.match(r'\s*RTC: [0-9\.]+ FPGA: ([0-9\.]+) dt=.*',dbg.message[idx])
        if m:
            last_fpga_status=dict(time_pre=parse_d(dbg.time.values[idx]),
                                  msg_pre=dbg.message[idx],
                                  fpga_pre=utils.unix_to_dt64(float(m.group(1))))
            idx+=1
            continue

        # 03/25/2019 11:04:45, FPGA started!! Init FPGA clock using RTC=1553540685.000000
        m=re.match(r'\s*FPGA started!! Init FPGA clock using RTC=([0-9\.]+)',dbg.message[idx])
        if m:
            # this message also sets the new, running status going forward
            new_status={}
            new_status['time_pre']=last_fpga_status['time_post']=parse_d(dbg.time.values[idx])
            new_status['msg_pre']=last_fpga_status['msg_post']=dbg.message[idx]
            new_status['fpga_pre']=last_fpga_status['fpga_post']=utils.unix_to_dt64(float(m.group(1)))
            clock_resets.append(last_fpga_status)
            last_fpga_status=new_status
            idx+=1
            continue

        # And pick up sync events
        if '-Before SYNC' not in dbg.message[idx]:
            idx+=1
            continue
        before_msg=dbg.message[idx]
        after_msg=dbg.message[idx+1]
        #  Things like
        #  03/13/2019 16:25:31, -Before SYNC: FPGA=1552523147.485166  RTC=1552523147.406250 dt=-0.078916
        #  03/13/2019 16:25:31, -After SYNC:  FPGA=1552523131.024603  RTC=1552523131.023437 dt=-0.001166
        assert '-After SYNC' in after_msg

        m1=re.search(r'FPGA=([0-9\.]+)',before_msg)
        m2=re.search(r'FPGA=([0-9\.]+)',after_msg)
        if m1:
            fpga_pre=utils.unix_to_dt64(float(m1.group(1)))
        else:
            fpga_pre=None
        if m2:
            fpga_post=utils.unix_to_dt64(float(m2.group(1)))
        else:
            fpga_post=None

        # dbg.time gives the timestamp of the log entry, but FPGA time is probably
        # what we actually want.
        clock_resets.append(dict(time_pre=parse_d(dbg.time.values[idx]),
                                 time_post=parse_d(dbg.time.values[idx+1]),
                                 msg_pre=dbg.message[idx],
                                 msg_post=dbg.message[idx+1],
                                 fpga_pre=fpga_pre,fpga_post=fpga_post))
        idx+=2

    clock_resets=pd.DataFrame(clock_resets)
    return clock_resets

def rs232_checksum(the_bytes):
    return '%02X' % (sum(the_bytes.encode()) & 0xFF)

def parse_txt(fn):
    """
    Parse a single txt formated Teknologic JSAT receiver file.
    Returns a pandas DataFrame, with many many fields
    """
    recs=[]
    
    with open(fn,'rt') as fp:
        while 1:
            if len(recs)>0 and len(recs)%1000==0:
                print(f"{len(recs)} records")
                
            rec={}
            
            line=fp.readline()
            if line=="":
                break
            else:
                line=line.strip()

            rec['raw']=line
                
            # Checksum:
            m=re.match('([^#]+),#([0-9A-F][0-9A-F])$',line)
            if m is not None:
                calc_sum=rs232_checksum(m.group(1))
                real_sum=m.group(2)
                if calc_sum==real_sum:
                    rec['checksum']=1
                else:
                    rec['checksum']=2
                line=m.group(1)
            else:
                rec['checksum']=-1

            # Try the known patterns:
            # NODE:001,187014,187014,,,SS-187014,20180328,1.098,224613,3748.450199,N,12119.655790,W,8.2,M,09,19.9,13.7,-105,56.9
            m=re.match('NODE:(.*)',line)
            if m is not None:
                rec['type']='NODE'
                csvs=m.group(1).split(',')
                (rec['node_serial'],
                 rec['rx_serial1'],
                 rec['rx_serial2'],
                 rec['dum1'],rec['dum2'],rec['rx_serial3'],rec['j_date'],rec['dum3'],rec['j_time'],
                 rec['lat_dm'],rec['lat_ns'],rec['lon_dm'],rec['lon_ew'],
                 rec['hdg'],rec['mag'],rec['dum4'],rec['dum5'],rec['dum6'],rec['dum7'],rec['dum8']) = csvs
            if m is None:
                #           serial  seq     date_str, STS, key=value
                m=re.match('[0-9]+,[0-9]+,([-0-9 :]+),STS,[,=A-Z0-9\.]+$',line)
                # 187014,000,2018-03-13 20:15:01,STS,FW=3.6.64,FPGA=830A,IVDC=4.0,EVDC=0.0,
                # I=0.023,VDC=17,IDC=0,DU=0,THR=100,BETA=0.760,NBW=28,AUTO=1,TILT=87.0,
                # PRESS=100801.4,WTEMP=23.4,ITEMP=19.1,#04
                if m is not None:
                    rec['type']='STATUS'
                    csvs=line.split(',')
                    (rec['rx_serial'],rec['seq'],rec['datetime_str']) = csvs[:3]
                    # parse comma-separated key=value pairs to dict:
                    rec.update( {p[0]:p[1] for p in [kv.split('=') for kv in csvs[4:]]} )
            if m is None:
                m=re.match('\*.*RTMNOW',line)
                if m is not None:
                    rec['type']='RTMNOW'
            if m is None:
                m=re.match('\*([0-9]+)\..*TIME=([-0-9 :]+)$',line)
                #'*187014.1#22,TIME=2018-03-13 20:14:54'
                if m is not None:
                    rec['type']='TIME'
                    rec['rx_serial']=m.group(1)
                    rec['datetime_str']=m.group(2)
            if m is None:
                m=re.match('\*([0-9]+)(\.\d+)[#0-9\[\]]+,(([\.0-9]+),([0-9]),)?OK',line)
                # Two types: 
                # *187016.0#23[0020],0.183095,0,OK,#BA
                #   potential clock resync. 
                # *<sn> not sure about these, then delta between two clocks in seconds,
                # then 0 if clock was left, 1 if it was resynced.
                # OK, and a checksum.
                # *187014.2#23[0009],OK,#9A
                #   comes right after an RTMNOW.
                if m is not None:
                    rec['rx_serial']=m.group(1)
                    if m.group(3) is not None:
                        rec['type']='SYNC'
                        rec['sync_dt']=float(m.group(4))
                        rec['sync_status']=int(m.group(5))
                    else:
                        rec['type']='OK' # prob. just boot up
            if m is None:
                m=re.match('[0-9]+,[0-9]+,([-0-9 :]+),[\.0-9]+,[0-9A-F]+,[0-9]+,[-0-9]+,[0-9]+,[0-9]+,[\.0-9]+',line)
                if m is not None:
                    rec['type']='DET'
                    csvs=line.split(',')
                    (rec['rx_serial'],rec['seq'],rec['datetime_str'],rec['t_usec'],
                     rec['tag_id'],rec['dum9'],rec['dum10'],rec['dum11'],
                     rec['dum12'],rec['dum13'])=csvs
            if m is None:
                rec['type']='unknown'
            
            recs.append(rec)

    df=pd.DataFrame(recs)
    df['time']=pd.to_datetime(df['datetime_str'])
            
    return df


def dice_by_clock_resets(ds,all_clock_resets):
    """
    ds: dataset, as from parse_tek()
    all_clock_resets: DBG-derived times when clock was changed, 
     as from dbg_to_clock_changes()
    returns: list of xr.Dataset
    """
    # Get rid of clock resets that are entirely before or entirely after
    # the pings
    t_min=ds.time.values.min()
    too_early=(all_clock_resets.fpga_pre<t_min)&(all_clock_resets.fpga_post<t_min)
    t_max=ds.time.values.max()
    too_late =(all_clock_resets.fpga_pre>t_max)&(all_clock_resets.fpga_post>t_max)

    out_of_bounds=too_early|too_late
    clock_resets=all_clock_resets[~out_of_bounds].copy()

    # Record which clock resets we've actually lined up with an index.
    # map index into clock_resets to an index into ds.index
    reset_to_break=-1*np.ones(len(clock_resets),np.int32)

    ds_times=ds.time.values
    breaks=[0]
    for i in range(len(ds.index)):
        # Keep going as long as it's possible for the samples to be part of the
        # current window.

        # Any backwards delta-t has to be a reset.
        # compare to ds_slice_start so 
        if (i>breaks[-1]) and (ds_times[i] < ds_times[i-1]):
            breaks.append(i)
            best_r_idx=None

            # And assign this to a clock reset
            for r_idx,rec in clock_resets.iterrows():
                if rec['fpga_pre'] < rec['fpga_pre']:
                    continue # only looking for backward jumps
                if ((rec['fpga_pre']>ds_times[i-1]) and
                    (rec['fpga_post']<ds_times[i])):
                    best_r_idx=r_idx
                    break
            if best_r_idx is None:
                # No exact match, so try allowing for inexact pre data.
                for r_idx,rec in clock_resets.iterrows():
                    if ( (rec['fpga_pre'] < rec['fpga_pre']) or
                         ('SYNC' in rec['msg_pre']) or
                         (reset_to_break[r_idx]>=0)):
                        continue # only looking for unmatched, inexact, backward jumps
                    if rec['fpga_post']<ds_times[i]:
                        # Possible, but now we want the *last* possible
                        # match
                        best_r_idx=r_idx
            assert best_r_idx is not None,"Really? Couldn't find a possible match?"
            reset_to_break[best_r_idx]=i
            continue

    if breaks[-1]<len(ds.index):    
        breaks.append(len(ds.index))

    # Group the clock resets by monotonic sequences
    known_resets=np.nonzero(reset_to_break>=0)[0]
    # Careful - this -1 actually means just before the beginning
    known_resets=np.r_[ -1,known_resets, len(clock_resets)]

    mono_slices=[] # [inclusive,exclusive]
    for a,b in zip(known_resets[:-1],
                   known_resets[1:]):
        if b-a>1:
            mono_slices.append( [a+1,b])

    for mono_a,mono_b in mono_slices:
        # Are the as-yet-unknown resets in this slice also monotonic?
        mono_resets=clock_resets.iloc[mono_a:mono_b,:]
        dt=np.diff(mono_resets['fpga_post'].values)
        if np.any(dt<0*dt):
            print('Warning: Non-monotonic fpga_post times within known block')
        if mono_a>0:
            break_before=reset_to_break[mono_a-1]
        else:
            break_before=0
        if mono_b<len(clock_resets):
            break_after=reset_to_break[mono_b]
        else:
            break_after=len(ds.index)
        # possible that break_before needs to be +1 here?
        breaks=break_before+np.searchsorted(ds_times[break_before:break_after],
                                            mono_resets['fpga_post'].values)
        reset_to_break[mono_a:mono_b]=breaks

    # Finally -- handle the actual dicing:
    assert np.all(np.diff(reset_to_break) >= 0 ),"Somehow have non-monotonicity"

    splits=np.r_[0,reset_to_break,len(ds.index)]
    diced=[]
    for a,b in zip(splits[:-1],splits[1:]):
        if a<b:
            diced.append( ds.isel(index=slice(a,b)) )
    return diced

