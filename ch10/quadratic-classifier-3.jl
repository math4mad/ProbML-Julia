"""
probml page 376 figure 10.7

mapping experiment space to feature space
Φ(x::Array)=[1, x[1], x[2], x[1]^2, x[2]^2, x[1]*x[2]]

MultinomialClassifier = @load MultinomialClassifier pkg=MLJLinearModels
多项式回归分类方法

BayesianQDA()  有 :`UserWarning: Variables are collinear
warnings.warn("Variables are collinear")` 问题

"""

import MLJ:predict_mode,fit!,predict

using MLJ, GLMakie,Random,DataFrames,ColorSchemes,Distributions,LinearAlgebra,KernelFunctions
#Random.seed!(123)
N=60

Φ(x)=[1.2, x[1], x[2], x[1]^2, x[2]^2, x[1]*x[2]]

"""
    create_data(N)
create  3 class mvnormal data
"""
function create_data(N)
    
    μs=[[0.5, 0.5],[-0.5, -0.5],[0.5, -0.5],[-0.5, 0.5],[0, 0]]
    cov = 0.01 * [1 0; 0 1]
    dists=[MvNormal(μ,cov) for μ in μs]
    X=hcat([rand(d,N) for d in dists]...)|>transpose
    newX=hcat([Φ(x) for x in eachrow(X)]...)|>transpose|>d->DataFrame(d,:auto)
    y=vcat([fill(i,N) for i in [1,1,2,2,3]]...)
    y= coerce(y,Multiclass)
    cats=levels(y)
    return newX,y,cats
end

function boundary_data(df;n=N)
    n1=n2=n
    xlow,xhigh=extrema(df[:,:x2])
    ylow,yhigh=extrema(df[:,:x3])
    tx = range(xlow,xhigh; length=n1)
    ty = range(ylow,yhigh; length=n2)
    x_test = hcat(mapreduce(collect, hcat, Iterators.product(tx, ty)))|>Matrix|>transpose
    newX=hcat([Φ(x) for x in eachrow(x_test)]...)|>transpose
    return tx,ty,newX
end

X,y,cats=create_data(N)
tx,ty,xtest=boundary_data(X)

function fit_model()
      
      MultinomialClassifier = @load MultinomialClassifier pkg=MLJLinearModels
      #BayesianQDA = @load BayesianQDA pkg=MLJScikitLearnInterface
      model1=MultinomialClassifier()
      #model2 = BayesianQDA()
      mach=machine(model1, X, y)|>fit!
     labels = predict_mode(mach, xtest)|>Array|>d->reshape(d,N,N)
     return labels
end

labels = fit_model()
function plot_res()
    
    fig = Figure(resolution=(800,400))
    ax1 = Axis(fig[1, 1], title="Origin data")
    ax2 = Axis(fig[1, 2], title="Quadratic classifier")
    cbarPal = :thermal
    cmap = cgrad(colorschemes[cbarPal], length(cats), categorical=true)
    for (idx, c) in enumerate(cats)
        local data = X[y.==c, :]
        scatter!(ax1, data[:,:x2], data[:, :x3]; color=(cmap[idx], 0.8), strokecolor=:black, strokewidth=1)
        scatter!(ax2, data[:, :x2], data[:, :x3]; color=(cmap[idx], 0.8), strokecolor=:black, strokewidth=1)
    end
    contourf!(ax2,tx,ty,labels;levels=length(cats),colormap=(:rainbow,0.2))
    
    save("./imgs/MultinomialClassifier.png",fig)
    fig
end
plot_res()







    
