"""
A brief foray into estimating the effect of noise on distance 
calculations.

In the mean, noise in the position estimates leads to an increase
in the average distance between position estimates.  We have
an estimate for the noise, and a single realization of the position
with noise.

Ideally we could adjust velocity distributions based on the s.d.
values from yaps, and avoid a high-bias in distances.

However, the actual errors are almost certainly not independent.
Error in two sequential positions is largely from error in clocks,
some geometry error, some measurement error, and some non-normal multipath
error.

Of those, the first two are persistent between pings, but then 
change when the hydrophone set changes. A proper accounting of these
errors would require going back to the yaps solution, or recreating
the yaps solution and sampling from the clock error states.

This is a place where a Bayesian approach would be nice since it would
give us a realizations of complete tracks which would be consistent
and give a reasonable velocity calculation.
"""

from scipy import stats
## 
# The 'truth'
dx=1.5
dy=2.0

D=dx**2+dy**2
d=np.sqrt(D)

# Noise samples
sx=1.0
sy=1.5

nsamp=50000
dx_n=stats.norm.rvs(dx,sx,nsamp)
dy_n=stats.norm.rvs(dy,sy,nsamp)

D_n=dx_n**2 + dy_n**2
d_n=np.sqrt(D_n)

plt.figure(12).clf()

fig,axs=plt.subplots(2,1,num=12)
axs[0].hist(D_n,bins=200)
axs[0].axvline(D_n.mean(),color='b',zorder=2)
axs[0].axvline(D,color='k',zorder=2)

axs[1].hist(d_n,bins=200)
axs[1].axvline(d,color='k',zorder=2)
axs[1].axvline(d_n.mean(),color='b',zorder=2)


##

dx=1.5
dy=0.0

D=dx**2+dy**2
d=np.sqrt(D)

nsamp=50000

# Noise samples
recs=[]
for sx in np.linspace(0,10,20):
    sy=0.0

    dx_n=stats.norm.rvs(dx,sx,nsamp)
    dy_n=stats.norm.rvs(dy,sy,nsamp)

    D_n=dx_n**2 + dy_n**2
    d_n=np.sqrt(D_n)

    recs.append( dict(sx=sx,D_n_mean=D_n.mean(),D=D,d=d,
                      d_n_mean=d_n.mean()) )

df=pd.DataFrame(recs)

plt.figure(13).clf()
fig,axs=plt.subplots(2,1,num=13)
axs[0].plot(df.sx,df.D_n_mean - D)
axs[0].plot(df.sx,df.sx**2) # YES

axs[1].plot(df.sx,df.d_n_mean - d)
axs[1].plot(df.sx,df.sx) # no - not so simple

# Am I close to reversing this?
# 
