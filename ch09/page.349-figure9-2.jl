
"""
probml page 349  figure9.2
"""

import MLJ:fit!,predict,predict_mode
using Distributions,GLMakie,Random,DataFrames,MLJ
Random.seed!(121212)
N_sample=30
xs=ys=range(-4.0,6.0,200)


Σ1 = [4 1; 1 2]
Σ2 = [2 0; 0 1]
Σ3 = [1 0; 0 1]
μs = [[0, 0], [0, 4], [4, 4]]

dists=[MvNormal(μs[1],Σ1),MvNormal(μs[2],Σ2),MvNormal(μs[3],Σ3)]

data=[rand(dist,N_sample) for dist in dists ]
X=hcat(data...)|>transpose
y=vcat(fill(1,N_sample),fill(2,N_sample),fill(3,N_sample))
y= coerce(y,Multiclass)

function make_testdata()
    n1=n2=200
    tx = range(-5,8,n1)
    ty = range(-5,8,length=n2)
    x_test = mapreduce(collect, hcat, Iterators.product(tx, ty));
    return tx,ty,x_test'
end
(tx,ty,x_test)=make_testdata()


#traing&predict
BayesianQDA = @load BayesianQDA pkg=MLJScikitLearnInterface
BayesianLDA = @load BayesianLDA pkg=MultivariateStats
modelQDA = BayesianQDA()
modelLDA = BayesianLDA()
mach1=machine(modelQDA, X, y)|>fit!
mach2=machine(modelLDA , X, y)|>fit!
QDAlabels = predict_mode(mach1, x_test)|>Array|>d->reshape(d,200,200)
LDAlabels = predict_mode(mach2, x_test)|>Array|>d->reshape(d,200,200)

function plot_res()
    fig=Figure(resolution=(1000,500))
    ax1=Axis(fig[1,1],title="QDA")
    ax2=Axis(fig[1,2],title="LDA")
    marker=[:circle,:rect,:cross]
    
    for (idx, d) in enumerate(dists)
        local zs=[pdf(d, [x, y]) for x in xs, y in ys]
        scatter!(ax1, data[idx][1,:],data[idx][2,:],marker=marker[idx])
        scatter!(ax2, data[idx][1,:],data[idx][2,:],marker=marker[idx])
        contour!(ax1,xs,ys,zs)
        contour!(ax2,xs,ys,zs)
    end
    contourf!(ax1,tx,ty,QDAlabels;levels=3,colormap=(:rainbow,0.2))
    contourf!(ax2,tx,ty,LDAlabels;levels=3,colormap=(:rainbow,0.2))
    #save("./imgs/page.349-figure9-2.png",fig)
    fig
end

plot_res()



