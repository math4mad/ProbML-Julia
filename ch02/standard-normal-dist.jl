using Distributions,GLMakie

fig = Figure(resolution=(1200,600))
ax1 = Axis(fig[1, 1],title=L"gaissan \quad cdf \qquad μ=0 \quad δ=1")
ax2 = Axis(fig[1, 2],title=L"gaissan \quad pdf \qquad μ=0 \quad δ=1")

d=Normal()


tspan=range(-4,4,200)
ps1=[cdf(d,t) for t in tspan]
ps2=[pdf(d,t) for t in tspan]

lines!(ax1, tspan, ps1)
lines!(ax2, tspan, ps2,color=:red)


fig
#save("./ch02/imgs/standard-normal-dist.png", fig)




