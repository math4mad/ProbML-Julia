
"""
probml page 349  figure9.1
"""

using Distributions,GLMakie,Random
Random.seed!(121212)
N_sample=30
xs=ys=range(-4.0,6.0,200)


Σ1 = [4 1; 1 2]
Σ2 = [2 0; 0 1]
Σ3 = [1 0; 0 1]
μs = [[0, 0], [0, 4], [4, 4]]

dists=[MvNormal(μs[1],Σ1),MvNormal(μs[2],Σ2),MvNormal(μs[3],Σ3)]

data=[rand(dist,N_sample) for dist in dists ]

fig=Figure(resolution=(1200,600))
ax1=Axis(fig[1,1],xlabel="(a)")
ax2=Axis(fig[1,2],xlabel="(b)")
marker=[:circle,:rect,:cross]

for (idx, d) in enumerate(dists)
    local zs=[pdf(d, [x, y]) for x in xs, y in ys]
    scatter!(ax1, data[idx][1,:],data[idx][2,:],marker=marker[idx])
    scatter!(ax2, data[idx][1,:],data[idx][2,:],marker=marker[idx])
    contour!(ax2,xs,ys,zs)
end

fig
#save("page-349-figure9-1.png",fig)


