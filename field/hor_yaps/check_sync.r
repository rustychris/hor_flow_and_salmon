library(yaps)
library(plotrix)
library("RColorBrewer")
library('tidyverse')
library('lubridate')

load('yaps/full/20180316T0152-20180321T0003/v04/sync_save')
sm<-sync_model

# TOP varies from 2.5 to 11059. one per ping.
# sync_model$inp_synced$dat_tmb_sync$toa_offset about the same.
# 
# there is a base unix epoch value 
# sync_model$inp_synced$inp_params$T0

plot(sm$pl$TOP)
sm$OFFSET
sm$SLOPE1
sm$SLOPE2

offset_levels<-sync_model$inp_synced$inp_params$offset_levels

# plot the offsets per hydro over several levels
#sel_offsets<-1:5
sel_offsets<-1:nrow(offset_levels)
T0<-sync_model$inp_synced$inp_params$T0

for( h in 1:13 ) {
  all_t<-c()
  all_o<-c()
  for ( i in sel_offsets ){
    t_start<-T0 + offset_levels[i,1]
    t_stop <-T0 + offset_levels[i,2]
    t_relative <- seq(from=0,to=offset_levels[i,2]-offset_levels[i,1],length.out=50)
    
    t_delta=t_relative/1e6 # maybe?
    OFF=sync_model$pl$OFFSET[h,i]
    S1=sync_model$pl$SLOPE1[h,i]
    S2=sync_model$pl$SLOPE2[h,i]
    
    offsets <- OFF + S1*t_delta + S2*t_delta*t_delta
    
    all_t<-c( all_t, t_relative + offset_levels[i,1] )
    all_o<-c( all_o, offsets)
  }
 
  sync_data<-dplyr::tibble(t=all_t,offset=all_o)
  label<-sync_model$inp_synced$inp_params$hydros$serial[h]

  # and the pings that were heard by this hydro
  toa_offset<-sync_model$inp_synced$dat_tmb_sync$toa_offset[,h]
  off_idx<-sync_model$inp_synced$dat_tmb_sync$offset_idx
  OFF<-sync_model$pl$OFFSET[h,off_idx]
  S1 <-sync_model$pl$SLOPE1[h,off_idx]
  S2 <-sync_model$pl$SLOPE2[h,off_idx]
  
  toa<-toa_offset + offset_levels[off_idx,1]
  t_delta <- toa_offset/1e6
  toa_off <- OFF + S1*t_delta + S2*t_delta*t_delta
  df_toa<-dplyr::tibble(toa=toa,off=toa_off)
  df_toa<-df_toa[ !is.na(df_toa$toa),]

  # And limit to selected offsets
  toa_min <- offset_levels[sel_offsets[1],1]
  toa_max <- offset_levels[max(sel_offsets),2]
  df_toa<-df_toa[ (df_toa$toa>=toa_min)&(df_toa$toa<=toa_max),]
  p<-ggplot(sync_data,aes(t,offset))+
    geom_line() +
    ggtitle(label) +
    geom_point(data=df_toa,aes(x=toa,y=off))
  print(p)
}
