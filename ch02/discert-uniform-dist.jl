"""
probml page65 figure2.1
"""

using  Distributions,GLMakie
a,b=1,4
d=DiscreteUniform(a,b)
fig=Figure()
ax=Axis(fig[1,1])
limits!(ax,0,5,0.0,1.0)
barplot!(ax,1:4,pdf(d))
fig
 #save("./imgs/discert-uniform-dist.png",fig)