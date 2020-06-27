# Try movehmm on a well-behaved track

require(moveHMM)
library(dplyr)
library(signal)
library(dplR)

# hand-pick tags with similar behaviors -- ferrying
# back and forth.
tag_ids<-c('7A96', '7ADB', '7B4D', '7B49', '7BD5', '7CA5', '7D53', '7D65','752B','758A',
          '769A','796D', '7567', '7585', '7895')
# tag_ids<-c('7A96')
##

remove_hydro <- function(track) {
  # reconstruct a series of positions that remove the
  # hydrodynamic velocity
  track_dt<- diff(track$tnum)
  #u<-track$swim_u[1:nrow(track)-1]
  #v<-track$swim_v[1:nrow(track)-1]
  u<-track$swim_urel[1:nrow(track)-1]
  v<-track$swim_vrel[1:nrow(track)-1]
  x0<-0 # track$x[1]
  y0<-0 # track$y[1]
  x<-x0 + c(0,cumsum(u*track_dt))
  y<-y0 + c(0,cumsum(v*track_dt))
  df<-data.frame(x=x,y=y,tnum=track$tnum)
  return(df)
}

expand <- function(track) {
  dt_nom=5.0 
  
  # Fill in missing samples by simple linear interpolation
  track_dt<- diff(track$tnum)
  track_nstep<-round(track_dt/dt_nom) # number of dt steps between positions
  n_expand<-1+sum(track_nstep)
  step_orig<-cumsum( c(1, track_nstep))
  step_expand<-1:n_expand
  x_expand <- approx(step_orig,track$x,xout=step_expand)
  y_expand <- approx(step_orig,track$y,xout=step_expand)
  df <- data.frame(x=x_expand$y, y=y_expand$y)
  return(df)
}

smooth <- function(df) {
  W<-4
  df$x<-pass.filt(df$x,W=8,type="low",n=2)
  df$y<-pass.filt(df$y,W=8,type="low",n=2)
  return(df)  
}

all_tracks<-NULL # with hydro velocity removed
all_trackhs<-NULL # with hydro velocity intact

for(tag_id in tag_ids) {
  trackh <- read.csv(paste('screen_final/track-',tag_id,'.csv',sep = ''))
  track <- smooth(expand(remove_hydro(trackh)))
  track$ID = tag_id
  all_tracks<-bind_rows(all_tracks,track)
  trackh <- smooth(expand(trackh))
  trackh$ID = tag_id
  all_trackhs<-bind_rows(all_trackhs,trackh)
}

data <- prepData(all_tracks,type="UTM")
#data$x <- all_trackhs$x
#data$y <- all_trackhs$y
# plot(data,compact=T,ask=FALSE)


##  ---- initial parameters for gamma and von Mises distributions
mu0 <- c(0.1,1) # step mean (two parameters: one for each state)
sigma0 <- c(0.1,1) # step SD
angleMean0 <- c(pi,0) # angle mean
kappa0 <- c(1,1) # angle concentration

## call to fitting function
m1 <- fitHMM(data=data,nbStates=2,
             stepPar0=c(mu0,sigma0),
             anglePar0=c(angleMean0,kappa0) )
plot(m1, plotCI=TRUE)

## ---- push for a slow ballistic, fast  ballistic, turning
# mu0 <- c(0.1,1.,0.1) # step mean
# sigma0 <- c(0.2,1,0.1) # step SD
# angleMean0 <- c(0,0,0) # angle mean
# kappa0 <- c(100,100,10) # angle concentration
# 
# ## call to fitting function
# m2 <- fitHMM(data=data,nbStates=3,
#              stepPar0=c(mu0,sigma0),
#              anglePar0=c(angleMean0,kappa0) )
# plot(m2, plotCI=TRUE)


# So what's up with, say 7CA5?
# state 1: slow-ish steps, narrow angle dist.
# state 2: large-ish steps, narrow angle dist
# state 3: short steps, wide angle dist.
# But 7CA5 
# plotStates(m2,animals="7CA5")
