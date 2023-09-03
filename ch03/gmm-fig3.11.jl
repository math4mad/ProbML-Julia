"""
probml page125  figure 3.11
"""

using Distributions,GLMakie

N = 60
xs=range(0,1 ; length=200)
ys=range(0,1; length=200)

function define_dist()
    w=[0.5 ,0.3 ,0.2]
    mu1 = [0.22, 0.45]
    mu2 = [0.5,0.5]
    mu3 = [0.77,0.55]

    sigma1 =[0.011  -0.01; -0.01  0.018]
    sigma2 =[0.018  0.01 ; 0.01  0.011]
    sigma3 = sigma1

    d1=MvNormal(mu1,sigma1)
    d2=MvNormal(mu2,sigma2)
    d3=MvNormal(mu3,sigma3)
    dists=[d1,d2,d3]
    mixturemodel = MixtureModel([d1,d2,d3],w)
    return (dists,mixturemodel)
end

(dists,mixturemodel)=define_dist()

zs = [pdf(mixturemodel, [x, y]) for x in xs, y in ys]

fig=Figure(resolution=(800,400))
    ax=Axis(fig[1,1],title="contour")
    ax2=Axis3(fig[1,2],title="mixture model")
    [contour!(ax,xs,ys,(x, y) -> pdf(d, [x, y])) for d in dists]
    surface!(ax2,xs,ys,zs)
fig
#save("./imgs/gaussian-mixture-model.png",fig)

