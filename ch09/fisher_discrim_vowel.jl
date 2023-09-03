"""
probml page356 fig9.5
"""

import MLJ:transform,fit!
using MLJ,DataFrames,MAT,GLMakie,Random,ColorSchemes
Random.seed!(34343)

data = matread("../data/vowelTrain.mat")
Xtrain=data["Xtrain"]|>d->DataFrame(d,:auto)
ytrain=data["ytrain"]|>d->coerce(d[:,1],Multiclass)

PCA = @load PCA pkg=MultivariateStats
LDA = @load LDA pkg=MultivariateStats
model = PCA(maxoutdim=2)
model2=LDA(;outdim=2)
mach = machine(model, Xtrain) |> fit!
mach2 = machine(model2, Xtrain,ytrain) |> fit!
Xproj1 = transform(mach, Xtrain)
Xproj2 = transform(mach2, Xtrain)

cats=levels(ytrain)

function plot_res()
    cbarPal = :thermal
    cmap = cgrad(colorschemes[cbarPal], length(cats), categorical = true)
    fig=Figure(resolution=(1000,500))
    ax1=Axis(fig[1,1],title="PCA")
    ax2=Axis(fig[1,2],title="LDA")

    for (idx,cat) in enumerate(cats)
        pca_data=Xproj1[ytrain.==cat,:]
        lda_data=Xproj2[ytrain.==cat,:]
        scatter!(ax1,pca_data[!,:x1],pca_data[!,:x2],color = cmap[idx])
        scatter!(ax2,lda_data[!,:x1],lda_data[!,:x2],color = cmap[idx])

    end
    fig
end

plot_res()