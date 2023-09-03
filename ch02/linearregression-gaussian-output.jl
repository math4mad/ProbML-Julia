"""
probml page 88 figure 2.14
"""

import  StatsFuns:softplus,log1psq
using Distributions,GLMakie,Random,StatsFuns
Random.seed!(22323)

xs=range(-20,60,200)
fμ(x)=1.2x
fσ(x)=3
fσ2(x)=log1psq(1.2x)

dist(x)=Normal(fμ(x),fσ(x)^2)
dist2(x)=Normal(fμ(x),fσ2(x)^2)


fig=Figure(resolution=(1200,600))

function plot_homoskedastic()
    ys=[rand(dist(x)) for x in xs]
    ax=Axis(fig[1,1],title="homoskedastic")
    scatter!(ax,xs,ys);
    lines!(ax, xs, fμ.(xs),color=(:red,0.8),linewidth=3)
    band!(ax,xs,fμ.(xs)-fill(9,200),fμ.(xs)+fill(9,200),color=(:green,0.3))
    
end



function plot_heteroskedastic()
    ys=[rand(dist2(x)) for x in xs]
    σ_arr=fσ2.(xs).|>d->d^2
    ax=Axis(fig[1,2],title="heteroskedastic")
    limits!(ax,-20,60,-100,100)
    scatter!(ax,xs,ys);
    lines!(ax, xs, fμ.(xs),color=(:red,0.8),linewidth=3)
    band!(ax,xs,fμ.(xs)-σ_arr,fμ.(xs)+σ_arr,color=(:green,0.3))
    fig
end

plot_homoskedastic()
plot_heteroskedastic()
#save("./imgs/lg-gaussian-output-2.png",fig)
