"""
probml page 376 figure 10.7
"""

import MLJ:predict_mode,fit!

using MLJ, GLMakie,Random,DataFrames,ColorSchemes,Distributions,LinearAlgebra
Random.seed!(123)
N=60

"""
    create_data(N)
create  3 class mvnormal data
"""
function create_data(N)
    μs=[[0.5, 0.5],[-0.5, -0.5],[0.5, -0.5],[-0.5, 0.5],[0, 0]]
    cov = 0.01 * [1 0; 0 1]
    dists=[MvNormal(μ,cov) for μ in μs]
    X=hcat([rand(d,N) for d in dists]...)|>transpose|>d->DataFrame(d,:auto)
    y=vcat([fill(i,N) for i in [1,1,2,2,3]]...)
    y= coerce(y,Multiclass)
    cats=levels(y)
    return X,y,cats
end

function boundary_data(df;n=200)
    n1=n2=n
    xlow,xhigh=extrema(df[:,:x1])
    ylow,yhigh=extrema(df[:,:x2])
    tx = range(xlow,xhigh; length=n1)
    ty = range(ylow,yhigh; length=n2)
    x_test = mapreduce(collect, hcat, Iterators.product(tx, ty));
    xtest=MLJ.table(x_test',names=[:x1,:x2])
    return tx,ty,xtest
end



X,y,cats=create_data(N)
tx,ty,xtest=boundary_data(X)  #decision boundary test  data


BayesianQDA = @load BayesianQDA pkg=MLJScikitLearnInterface
modelQDA = BayesianQDA()
mach=machine(modelQDA, X, y)|>fit!
labels = predict_mode(mach, xtest)|>Array|>d->reshape(d,200,200)

function plot_res()
    fig = Figure(resolution=(800,400))
    ax1 = Axis(fig[1, 1], title="Origin data")
    ax2 = Axis(fig[1, 2], title="Quadratic classifier")
    cbarPal = :thermal
    cmap = cgrad(colorschemes[cbarPal], length(cat), categorical=true)
    for (idx, c) in enumerate(cats)
        local data = X[y.==c, :]
        scatter!(ax1, data[:, :x1], data[:, :x2]; color=(cmap[idx], 0.8), strokecolor=:black, strokewidth=1)
        scatter!(ax2, data[:, :x1], data[:, :x2]; color=(cmap[idx], 0.8), strokecolor=:black, strokewidth=1)
    end
    contourf!(ax2,tx,ty,labels;levels=length(cats),colormap=(:rainbow,0.2))
    
    #save("./imgs/page-376-figure-10-7.png",fig)
    fig
end


plot_res()
