using MLJ, DataFrames,Plots
import MLJ:predict_mode
X, y = make_moons(100; noise=0.05)
X = DataFrame(X)


"""
    grid_data(col1,col2)
根据 两列数据生成绘制决策边界的xspan,yspan 和 y_test 数据
覆盖x,y 坐标的整个范围

return  gridx,gridy,x_test

"""
function grid_data(col1,col2)
n1 = n2 = 200
xlow, xhigh = extrema(col1)
ylow, yhigh = extrema(col2)
gridx = range(xlow, xhigh; length=n1)
gridy = range(ylow, yhigh; length=n2)
x_test = mapreduce(collect, hcat, Iterators.product(tx, ty))|>transpose|>MLJ.table
return gridx,gridy,x_test
end

gridx,gridy,x_test=grid_data(X[:,1],X[:,2])




KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels
model = KNNClassifier(weights = NearestNeighborModels.Inverse(),algorithm=:kdtree)
mach = machine(model,X,y)|> fit!

#ypred = predict(mach, x_test)
test_labels = predict_mode(mach, x_test)

cat =  y|> levels |> length

#决策边界图
contourf(gridx,gridy, test_labels, levels=cat, color=cgrad(:roma, 10, categorical = true, scale = :exp), alpha=0.7)

# 原始数据图,变量坐标系中所有 x,y 然后做出预测
scatter!(X[:, :x1], X[:, :x2], group=y, label=false, ms=3, alpha=0.3)

#savefig("moon-knearest-neighbour.png")