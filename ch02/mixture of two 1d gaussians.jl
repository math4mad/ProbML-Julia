"""
probml page 40 figure2.4
"""

using Distributions,GLMakie

fig = Figure(resolution=(800,1800))
ax1 = Axis(fig[1, 1],title=L"gaissan  \qquad μ=0 \quad δ=0.5")
ax2 = Axis(fig[2, 1],title=L"gaissan   \qquad μ=2 \quad δ=0.5")
ax3 = Axis(fig[3, 1],title=L"two gaissan 1d  mixutures" )

d1=Normal(0,0.5)
d2=Normal(2,0.5)
tspan=range(-4,4,200)

ps1=[pdf(d1,t) for t in tspan]
ps2=[pdf(d2,t) for t in tspan]
ps3=[pdf(d1,t)+pdf(d2,t) for t in tspan]

lines!(ax1, tspan, ps1;color=:green)
lines!(ax2, tspan, ps2;color=:red)
lines!(ax3, tspan, ps3)


fig
#save("mixture of two 1d gaussians.png", fig)

