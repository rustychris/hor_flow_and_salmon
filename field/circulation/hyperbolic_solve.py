import sympy
from sympy.solvers import solve
from sympy import Symbol,sqrt

x= Symbol('x')
y=Symbol('y')

delta_dist=10.0

rx0=np.array([10,10])
rx1=np.array([20,0])
rx2=np.array([0,0])
xy=np.array([5,10])

dist0_sq=(rx0[0]-x)**2 + (rx0[1]-y)**2
dist1_sq=(rx1[0]-x)**2 + (rx1[1]-y)**2
dist2_sq=(rx2[0]-x)**2 + (rx2[1]-y)**2

d01=utils.dist(rx0,xy) - utils.dist(rx1,xy)
d12=utils.dist(rx1,xy) - utils.dist(rx2,xy)

eqs=[sqrt(dist0_sq) - sqrt(dist1_sq) - d01,
     sqrt(dist1_sq) - sqrt(dist2_sq) - d12]

# This takes... 1 second
result=solve(eqs,set=True) 
print( result)
# => [5,10],[-51,169]

##

# Any chance it can find the solution symbolically with respect
# to the time deltas?  If so, I can pre-process combinations
# of stations to closed form, and quickly solve for different
# pings.
td0=Symbol('t_0')
td1=Symbol('t_1')

eqs=[sqrt(dist0_sq) - sqrt(dist1_sq) - td0,
     sqrt(dist1_sq) - sqrt(dist2_sq) - td1]

# That fails to find a solution
result=solve(eqs,[x,y],set=True)

## 
import matplotlib.pyplot as plt
plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

rxs=np.array([rx0,rx1,rx2])
ax.plot(rxs[:,0],rxs[:,1],'go')
ax.plot([xy[0]],[xy[1]],'ro')

output=np.array(list(result[1]))
ax.plot(output[:,0],output[:,1],'.',color='b',zorder=-2,ms=20)

ax.axis('equal')

# Can I plot the hyperbolas themselves?
all_res=[]

for a,b in [(x,y),
            (y,x)]:
    for eq,col in zip(eqs,
                      ['k','0.5']):
        for res in solve(eq,a): # solve for a, 
            bi=np.linspace(-200,200,600)
            solns=[]
            for bb in bi:
                aval=res.subs(b,bb)
                if aval.is_real:
                    aval=float(aval)
                    if a==x:
                        solns.append( [aval,bb] )
                    else:
                        solns.append( [bb,aval] )
                else:
                    solns.append( [np.nan,np.nan] )
            solns=np.array(solns)

            #ax.plot(solns[:,0],solns[:,1],'.',color=col)
            ax.plot(solns[:,0],solns[:,1],'-',color=col)

# This checks out. Nice.


##
