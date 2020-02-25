"""
Read receiver data, split into manageable chunks (~6h), and check
for clock resets (which are convered to two independent stations)

Pass to R/yaps for fitting clock drift model.
"""
import datetime
import numpy as np
import re
from stompy import utils
import matplotlib.pyplot as plt
import prepare_pings
import parse_tek as pt
import pandas as pd
import six
## 
six.moves.reload_module(pt)
## 
# pm=prepare_pings.ping_matcher_2018(split_on_clock_change=True)

am1_ds=pt.parse_tek(name="AM1",det_fn='2018_Data/AM1_20186009/050318/AM1180711347.DET',
                 pressure_range=None,
                 split_on_clock_change=False)

# Find a shore station that has some significant resets.

# two very minor clock adjustments
sm1_ds=pt.parse_tek(name='SM1',det_fn='2018_Data/SM1_20187013/050218/SM1180711252.DET',
                    pressure_range=None,split_on_clock_change=False)

# one very small change
sm2_ds=pt.parse_tek(name='SM2',det_fn='2018_Data/SM2_20187014/050218/SM2180711257.DET',
                    pressure_range=None,split_on_clock_change=False)

sm3_ds=pt.parse_tek(name='SM3',det_fn='2018_Data/SM3_20187015/SM3180711301.DET',
                    pressure_range=None,split_on_clock_change=False)

sm4_ds=pt.parse_tek(name='SM3',det_fn='2018_Data/SM4_20187018/050218/SM6180711928.DET',
                    pressure_range=None,split_on_clock_change=False)

sm8_ds=pt.parse_tek(name='SM3',det_fn='2018_Data/SM8_20187017/050218/SM5180711921.DET',
                    pressure_range=None,split_on_clock_change=False)

sm9_ds=pt.parse_tek(name='SM3',det_fn='2018_Data/SM9_20187024/050218/SM12180721837.DET',
                    pressure_range=None,split_on_clock_change=False)

## 
# 2019 SM1 had a lot of resets, I think
sm1_2019_ds=pt.parse_tek(name="SM1_19",
                         det_fn="2019 Data/HOR_Flow_TekDL_2019/SM1_187023/20190517/187023_190416_050506_P/sm18-7023190711049.DET",
                         split_on_clock_change=True)

## 
#diced=dice_by_clock_resets(ds,all_clock_resets)
diced=sm1_2019_ds

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)
for dds in diced:
    ax.plot( dds.index.values, dds.time.values, '.')

