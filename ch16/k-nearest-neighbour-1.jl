using CSV
using DataFrames
using MLJ
using Plots
import MLJ:predict,predict_mode

urls(str) = "/Users/lunarcheung/Public/DataSets/clustering-datasets/$str.csv"
f(str) = urls(str) |> CSV.File |> DataFrame

df=f("basic1")
rows, _ = size(df)
x1 = df[1:2:rows, :x]
x2 = df[1:2:rows, :y]
X = hcat(x1, x2)
y = df[1:2:rows, :color]
X = MLJ.table(X)
y = categorical(y)


#test
n1 = n2 = 200
xlow, xhigh = extrema(df[:, :x])
ylow, yhigh = extrema(df[:, :y])
tx = range(xlow, xhigh; length=n1)
ty = range(ylow, yhigh; length=n2)
x_test = mapreduce(collect, hcat, Iterators.product(tx, ty))
x_test = MLJ.table(x_test')


KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels
model = KNNClassifier(weights = NearestNeighborModels.Inverse(),algorithm=:kdtree)
mach = machine(model,X,y)|> fit!

ypred = predict(mach, x_test)
labels = predict_mode(mach, x_test)

cat = df[:, :color] |> levels |> length

#决策边界图
contourf(tx, ty, labels, levels=cat, color=cgrad(:heat), alpha=0.7)

# 原书数据图,变量坐标系中所有 x,y 然后做出预测
scatter!(df[:, :x], df[:, :y], group=df[:, :color], label=false, ms=3, alpha=0.3)


#savefig("k-nearest-neighbour-2.png")





