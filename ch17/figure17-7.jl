"""
probml page 660  figure17.7
"""

using KernelFunctions,GLMakie,Random,LinearAlgebra,Distributions
using GaussianProcesses,Optim
Random.seed!(2343)

k=SqExponentialKernel()
xs=[-4.0,-3.0, -2.0,-1.0,1.0]
ys=sin.(xs)

fig=Figure(resolution=(1200,300))
ax=[Axis(fig[1,i]) for i in 1:4]

function prior_sample(ax)
    local xrange=range(-5,5,100)
    K = kernelmatrix(k, xrange)
    f=rand(MvNormal(K+1e-6*I),6)|>transpose
    series!(ax,xrange,f,linewidth=2)
    ax.title="prior samples"
end


function posterior_sample(ax,spots=2)
    mZero = MeanZero()
    kern = SE(0.0, 0.0)
    logObsNoise = -1.0
    gp = GP(xs[1:spots], ys[1:spots], mZero, kern)
    optimize!(gp; method=ConjugateGradient())
    μ, σ²=predict_y(gp,range(-5,5,100));
    #prior_sample(ax)
    
    samples = rand(gp, range(-5,5,100), 7)|>transpose
    series!(ax,range(-5,5,100),samples)
    scatter!(ax,xs[1:spots], ys[1:spots];marker=:circle,markersize=14,color=(:lightgreen,0.2),strokewidth=2,strokecolor=:black)
    band!(ax,range(-5,5,100),μ+sqrt.(σ²),μ-sqrt.(σ²),color=(:red,0.5))
    ax.title="posterior samples $(spots) points"
    ax.limits=(-5,5,-2,2)
end

prior_sample(ax[1])
posterior_sample(ax[2],2)
posterior_sample(ax[3],4)
posterior_sample(ax[4],5)
fig
#save("./imgs/figure17-7-3-with-optim.png",fig)
