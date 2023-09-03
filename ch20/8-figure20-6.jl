"""
from probml page 689 figure20-8
"""

import MLJ:transform,inverse_transform
using MLJ,DataFrames,GLMakie,CSV,Random
fontsize_theme = Theme(fontsize = 24)
set_theme!(fontsize_theme)
Random.seed!(12121)

#data processing

function data_prepare(str)
    fetch(str) = str |> d -> CSV.File("./data/$str.csv") |> DataFrame |> dropmissing
    to_ScienceType(d)=coerce(d,:label=>Multiclass)
    df = fetch(str)|>to_ScienceType
    return df
end

str1="mnist_train"
#str2="mnist_test"
data=data_prepare(str1)
ytrain,Xtrain=  unpack(data, ==(:label), rng=123);


PCA = @load PCA pkg=MultivariateStats
trainerror=[]
testerror=[]
function train_model()
    for maxdim in [1,3,10,50:50:400...]
        model=PCA(maxoutdim=maxdim)
        mach = machine(model, Xtrain) |> fit!
        Xtr =transform(mach, Xtrain)
        rec_Xtrain=inverse_transform(mach,Xtr)
        resdiue1=rms(Xtrain|>Matrix, rec_Xtrain|>Matrix)|>d->round(d,digits=3)
        push!(trainerror,resdiue1)
    end
end

fig=Figure(resolution=(300,300))
ax=Axis(fig[1,1],xlabel="num PCs",ylabel="mse")
scatterlines!(ax,[1,3,10,50:50:400...],trainerror.|>Float32, marker=:circle, markersize=10, strokewidth=1, strokecolor=:black,color=(:lightblue,0.6))
#save("mnist-pca-mse.png",fig)

