# follow m-clark.github.io intro, use mgcv
library(mgcv)
library(ggplot2)
library(GGally)
library(visreg)
library(dplyr)
library(gridExtra)
library(itsadug)

# raw data as output by track_analyses.py
seg_data <- read.csv('../hor_yaps/segment_correlates.csv')
# convert tag id to a factor for use as random effect in mgcv
seg_data$tag <- factor(seg_data$id)
seg_data$waterdepth <- seg_data$model_z_eta - seg_data$model_z_bed
seg_data$vor <- abs(seg_data$model_vor_top2m)

cols <- c("tag","swim_urel","swim_lat","hydro_speed","turb","hour","flow_100m3s",
          "vor","waterdepth","tnum")

mod_data <- start_event(seg_data[cols],"tnum",event="tag")


# GAM for instantaneous rheotaxis
# Most of the correlates do not have meaningful variation within a 
# track, for example mean river flow does not change over the course
# of a 20 minute track. Here we lump all between-tag variance in a per-tag
# random effect, and model the remaining within-track variance.
# [mgcv with random effects following notes at
#  https://fromthebottomoftheheap.net/2021/02/02/random-effects-in-gams/ ]

# Initial model with all potential terms included
mod_segs <- gam(swim_urel ~ s(hydro_speed,bs="cr") 
                 + s(swim_lat,bs="cr")
                 + s(vor,bs="cr")
                 + s(waterdepth,bs="cr")
                 + s(tag,bs='re'), 
                 data=mod_data)
#summary(mod_segs)
#visreg(mod_segs)
# concurvity(mod_segs,full=FALSE)
# concurvity(mod_segs,full=TRUE)
# BIC(mod_segs) # -5128

# Following tips here:
# https://cran.r-project.org/web/packages/itsadug/vignettes/acf.html
# These are in fact all identical
acf(resid(mod_segs), main="acf(resid(mod_segs))")
#acf(resid_gam(mod_segs), main="acf(resid_gam(mod_segs))")
#acf_resid(mod_segs, main="acf_resid(mod_segs)")
r1 <- start_value_rho(mod_segs, plot=TRUE)

# Seems I have to use bam in order for the AR options 
# to take effect. There are still some outliers in swim_lat that 
# are potentially forcing it to be more important than it really is.
# Brute-force removal of these outliers did not substantially change
# p-values. 
# Limiting knots to 3 also did not reduce significance, but made
# the fits smoother.
mod_segs_ar <- bam(swim_urel ~ s(hydro_speed,bs="cr") 
                + s(swim_lat,bs="cr")
                + s(vor,bs="cr")
                + s(waterdepth,bs="cr")
                + s(tag,bs='re'), 
                rho=r1, AR.start=mod_data$start.event,
                data=mod_data,
                gamma=10)
summary(mod_segs_ar)
options(repr.plot.width=6,repr.plot.height=5.5)
p1<-visreg(mod_segs_ar,"swim_lat",fill=list(fill="green"), points=list(alpha=0.4),gg=TRUE)
p2<-visreg(mod_segs_ar,"hydro_speed",fill=list(fill="green"),points=list(alpha=0.4),gg=TRUE)
p3<-visreg(mod_segs_ar,"waterdepth",fill=list(fill="green"),points=list(alpha=0.4),gg=TRUE)
p4<-visreg(mod_segs_ar,"tag",gg=TRUE)
pan<-grid.arrange(p1,p2,p3,p4,nrow=2)
ggsave('mod_segs_lon.png',plot=pan,width=6,height=5.5)

# summary reports that everything but vorticity is
# significant. Visual plots suggest that swim_lat is
# not significant, though.


# Within-track Lateral swimming
mod_seglat <- gam(swim_lat ~ s(hydro_speed,bs="cr") 
                + s(swim_urel,bs="cr")
                + s(vor,bs="cr")
                + s(waterdepth,bs="cr")
                + s(tag,bs='re'), 
                data=mod_data)
acf(resid(mod_seglat), main="acf(resid(mod_seglat))")
r1_lat <- start_value_rho(mod_seglat, plot=TRUE)

mod_seglat_ar <- bam(swim_lat ~ s(hydro_speed,bs="cr") 
                   + s(swim_urel,bs="cr")
                   + s(vor,bs="cr")
                   + s(waterdepth,bs="cr")
                   + s(tag,bs='re'), 
                   rho=r1_lat, AR.start=mod_data$start.event,
                   family=Gamma,
                   data=mod_data)
summary(mod_seglat_ar)
visreg(mod_seglat_ar,scale="response") # ,partial=TRUE)

# # Sub-sampling to understand effects of autocorrelation
# # When sub-sampling, the fits are twitchy, even when 
# # forcing a simpler model with gamma.
# for ( it in 1:10 ) {
#   mod_data_sub <- sample_n(mod_data,as.integer(nrow(mod_data)/2)) 
#   mod_sub_segs <- gam(swim_urel ~ s(hydro_speed,bs="cr") 
#                   + s(swim_lat,bs="cr")
#                   + s(vor,bs="cr")
#                   + s(waterdepth,bs="cr")
#                   + s(tag,bs='re'), 
#                   data=mod_data_sub,gamma=15)
#   #summary(mod_sub_segs)
#   # Just the p-values for the smooth terms.
#   print(summary(mod_sub_segs)$s.table[,4])
# }
# Regardless of gamma, seems the p-values are not stable.


# With no weighting, n=6457
# and all terms are highly significant.
# Changing weights decreased fREML, but did not change
# anything else.
# Gamma makes a difference. It decreases edf from 4.5 and 6
# to 2 and 4. Unfortunately this also decreases the strength
# of the random effects term, decreasing its edf from 105 to 20.
# weights<-rep(0.01,nrow(mod_data))

## 

# Summarize the data per track and fit 
track_data <- seg_data %>% 
              group_by(tag) %>% 
              summarize(swim_lat=mean(swim_lat),
                        urel=mean(swim_urel),
                        swim_lon=mean(abs(swim_urel)),
                        hour=min(hour),
                        turb=mean(turb),
                        speed=mean(hydro_speed),
                        flow=mean(flow_100m3s),
                        reach_velo=mean(reach_velo_ms))

m_lon <- gam(urel ~ s(speed,bs="cr") + s(hour,bs="cc") + s(turb,bs='cr') +
                    s(flow,bs='cr') + s(reach_velo,bs='cr'), 
                data=track_data,knots=list(hour=c(0,24)))
summary(m_lon)
visreg(m_lon)

# hour and flow are near significant at 0.06 and 0.056
# Recall this is for urel!
m_lon_slim <- gam(urel ~ s(hour,bs="cc") + s(flow,bs='cr'), 
             data=track_data,knots=list(hour=c(0,24)))
summary(m_lon_slim)
visreg(m_lon_slim)

#track_data$swim_lon=
  
# this shows hour significant (p<1e-6), and flow (p=0.001), but
# not reach_velo, turb, or mean hydro speed.
# and flow and reach_velo are sort of fighting each other.
# mean hydro speed seems to say nothing. Dropping turb, reach_velo and
# hydro speed, and flow becomes more significant, at p=0.0004.
# however the shape of the flow curve is not very convincing.
# pretty good:
#m_lat <- gam(swim_lat ~ s(hour,bs="cc") + s(flow,bs='cr'), 
#             data=track_data,knots=list(hour=c(0,24)))

# a second term of flow, turb or urel are all signficant on their own 
# and flow+urel is pretty strong.
# seems that turbidity and flow are concurved, with turbidity edged out 
# with gamma around 5, flow becomes uninformative.
m_lat <- gam(swim_lat ~ s(hour,bs="cc") + s(urel,bs='cr') + s(flow,bs='cr'), 
             data=track_data,knots=list(hour=c(0,24)),
             gamma=5)
summary(m_lat)
visreg(m_lat,'flow')
# 

# log link function -- doesn't change results much from above.
m_loglat <- gam(log(swim_lat) ~ s(hour,bs="cc") + s(urel,bs='cr') + s(flow,bs='cr'), 
             data=track_data,knots=list(hour=c(0,24)),
             gamma=5)
summary(m_loglat)
visreg(m_loglat,scale="response",partial=TRUE)

# Attempting to use a Gamma distribution with log-link in order
# to acknowledge the non-negative nature of swim_lat.
# The coefficients in the fit are now sort of crazy
# The plots make it look like it fits really well, though.  Too well?
# The overall predictions look believable, though.
# Part of this is probably because the link function is inverse, so I
# think I'm seeing the inverse of the true predictors.
# 
m_lat_gamma <- gam(swim_lat ~ s(hour,bs="cc") + s(urel,bs='cr') + s(flow,bs='cr',k=3), 
                   data=track_data,knots=list(hour=c(0,24)),
                   family=Gamma)
summary(m_lat_gamma)
# plot on response scale to keep it more physical
# must explicitly request partial plot, 
# https://github.com/pbreheny/visreg/issues/55
# caveat emptor -- residuals may be misleading in
# ways I don't understand
visreg(m_lat_gamma,scale="response",partial=TRUE)

m_lat_invgss <- gam(swim_lat ~ s(hour,bs="cc") + s(urel,bs='cr') + s(flow,bs='cr',k=3), 
                   data=track_data,knots=list(hour=c(0,24)),
                   family=inverse.gaussian)
summary(m_lat_invgss)
visreg(m_lat_invgss,scale="response",partial=TRUE)

# These plots with partial=True I do think are misleading -- something
# is off with the scale of the residuals.




# Full scatter on the result:
fits <- predict(m_lat_gamma,type="response")
data <- track_data
data$fit_swim_lat <- fits

plt <- ggplot(data=data,aes(x=swim_lat,y=fit_swim_lat))+ geom_point()

# r-squared of 0.36 (not too different from the summary R-sq(adj) of 0.32)
cor(fits,track_data$swim_lat)^2

# "Histomancy" as McElreath calls it.  Plot a histogram of swim_lat and stare
# at it to guess the distribution
ggplot(data=data,aes(x=swim_lat)) + 
  geom_histogram() +
  geom_density(alpha=0.2,fill='#8888ee')
# That looks left-skewed

# Better, maybe, is to look at the distribution of residuals
# Not that telling.  Fairly compact distribution around 0.
# We know that there is a lower bound on swimm
ggplot(data=data.frame(res=fits-track_data$swim_lat),
       aes(x=res)) + geom_histogram()
