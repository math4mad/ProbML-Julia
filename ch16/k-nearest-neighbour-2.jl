"""
using Mahalanobis distance
"""


using CSV
using DataFrames
using MLJ
using Plots
using KernelFunctions
import MLJ:predict,predict_mode
using   Distances

urls(str) = "/Users/lunarcheung/Public/DataSets/clustering-datasets/$str.csv"
f(str) = urls(str) |> CSV.File |> DataFrame

df=f("basic1")
rows, _ = size(df)
x1 = df[1:2:rows, :x]
x2 = df[1:2:rows, :y]
#X = hcat(x1, x2)|>transpose
#Xtable = MLJ.table(X)
y = df[1:2:rows, :color]
y = categorical(y)
#k = SqExponentialKernel()
X=df[1:1:end,1:2]
#kmatrix=kernelmatrix(k, RowVecs(X))



function  create_boundary_data(df)
    #test
    n1 = n2 = 200
    xlow, xhigh = extrema(df[:x1])
    ylow, yhigh = extrema(df[:x2])
    tx = range(xlow, xhigh; length=n1)
    ty = range(ylow, yhigh; length=n2)
    x_test = mapreduce(collect, hcat, Iterators.product(tx, ty))
    x_test = MLJ.table(x_test')
end
#(tx,ty,x_test)=create_boundary_data(Xtable)


# KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels

# model = KNNClassifier(;algorithm =:brutetree,weights = NearestNeighborModels.Inverse(),metric=Distances.Mahalanobis(kmatrix))

KNeighborsClassifier = @load KNeighborsClassifier pkg=MLJScikitLearnInterface
model = KNeighborsClassifier(;metric=mahalanobis)

mach = machine(model,X,y)|> fit!   

#ypred = predict(mach, x_test)
#labels = predict_mode(mach, x_test)

cat = df[:, :color] |> levels |> length

#决策边界图
#contourf(tx, ty, labels, levels=cat, color=cgrad(:heat), alpha=0.7)

# 原书数据图,变量坐标系中所有 x,y 然后做出预测
#scatter!(df[:, :x], df[:, :y], group=df[:, :color], label=false, ms=3, alpha=0.3)


#savefig("k-nearest-neighbour-Mahalanobis.png")

