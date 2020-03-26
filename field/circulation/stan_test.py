import pystan

model1="""
data {
 real x;
}
parameters {
 real s;
}
model{
 x ~normal(2*s,1);
}
"""
sm1 = pystan.StanModel(model_code=model1)

with open('model1.cpp','wt') as fp:
    fp.write(sm1.model_cppcode)
## 

model2="""
data {
 real x;
}
parameters {
 real s;
}
model{
 s ~normal(x/2,1);
}
"""
sm2 = pystan.StanModel(model_code=model2)
with open('model2.cpp','wt') as fp:
    fp.write(sm2.model_cppcode)

##

# If I did want to include the jacobian..    
# dt[a]=sqrt( square(rx_x[a] - x) + square(rx_y[a]-y)) / rx_c[a];

# tdoa=rx_t[b]-rx_t[a];
# tdoa ~ normal(dt[b]-dt[a],sigma_t);



from sympy import *
x,y,rx1,ry1,rx2,ry2,rx3,ry3=symbols("x,y,rx1,ry1,rx2,ry2,rx3,ry3")

dt1=sqrt( (x-rx1)**2 + (y-ry1)**2)
dt2=sqrt( (x-rx2)**2 + (y-ry2)**2)
dt3=sqrt( (x-rx3)**2 + (y-ry3)**2)

delta12=dt1-dt2
delta13=dt1-dt3
delta23=dt2-dt3


# my parameters are x,y
# but they get transformed to delta12,delta13,delta23
# so 2 parameters, but 3 transformed parameters
# say I limit myself to just delta12, delta13

dist1,dist2,dist3=symbols('dist1,dist2,dist3')

Jac=Matrix([ [ diff(delta12,x), diff(delta12,y)],
             [ diff(delta13,x), diff(delta13,y)]] )
Jac=Jac.subs(dt1,dist1).subs(dt2,dist2).subs(dt3,dist3)

Jac.det()

#  ⎛  -rx₂ + x   -rx₁ + x⎞ ⎛  -ry₃ + y   -ry₁ + y⎞   ⎛  -ry₂ + y   -ry₁ + y⎞ ⎛  -rx₃ + x   -rx₁ + x⎞
#  ⎜- ──────── + ────────⎟⋅⎜- ──────── + ────────⎟ - ⎜- ──────── + ────────⎟⋅⎜- ──────── + ────────⎟
#  ⎝   dist₂      dist₁  ⎠ ⎝   dist₃      dist₁  ⎠   ⎝   dist₂      dist₁  ⎠ ⎝   dist₃      dist₁  ⎠

# ((x-rx1)/dt1 - (x-rx2)/dt2) * ( (y-ry1)/dist1 - (y-ry3)/dist3) -
#    ( (y-ry1)/dist1 - (y-ry2)/dist2) * ( (x-rx1)/dist1 - (x-rx3)/dist3 )

# Still a bit removed from the unit hyperbola 1/L formula.
# 

