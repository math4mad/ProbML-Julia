"""
probml page62 figure 2.17

Β 分布
"""

using GLMakie,Distributions

struct Bparams
    a
    b
end

p1=Bparams(0.1,0.1)
p2=Bparams(0.1,1.0)
p3=Bparams(1.0,1.0)
p4=Bparams(2.0,2.0)

p5=Bparams(2.0,8.0)

#d1=Beta(p1.a,p1.b)

p4



