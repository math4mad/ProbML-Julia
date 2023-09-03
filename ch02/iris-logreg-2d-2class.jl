"""
probml page 84,figure 2.13
概率值预测方法 参见: https://discourse.julialang.org/t/extracting-values-from-univariatefinite/62794/3
"""

import MLJ:fit!,predict,predict_mode,predict_mean
using MLJ,GLMakie,Random,DataFrames

iris = load_iris(); 
iris =DataFrame(iris);
nums=100

iris[!, :target] = [r.target == "virginica" ? 1.0 : 0.0 for r in eachrow(iris)]
iris=coerce(iris, :target=> Multiclass )
gdf=groupby(iris, :target)
X,y=iris[:,3:4],iris[:,:target]

cats=levels(y)

function boundary_data(df,;n=nums)
    n1=n2=n
    xlow,xhigh=extrema(df[:,1])
    ylow,yhigh=extrema(df[:,2])
    tx = LinRange(xlow,xhigh,n1)
    ty = LinRange(ylow,yhigh,n2)
    x_test = mapreduce(collect, hcat, Iterators.product(tx, ty));
    x_test=MLJ.table(x_test')
    return tx,ty,x_test
end
tx,ty,x_test=boundary_data(X)

"""
    eval_model()
## 实例化分类模型, 返回预测标签和1.0标签的预测概率
`return yhat,probs_res`

math4mads
"""
function eval_model()
    LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels
    mach = fit!(machine(LogisticClassifier(), X, y))
    yhat=predict_mode(mach, x_test)|>Array|>d->reshape(d,nums,nums)  #返回分类标签
    probs=predict(mach, x_test)|>Array #返回分类概率值
    probs_res=broadcast(pdf, probs, 1.0).|>(d->round(d,digits=2))|>d->reshape(d,nums,nums) #返回概率为1.0("virginica")的概率值
    return yhat,probs_res
end

yhat,probs_res=eval_model()

fig=Figure(resolution=(1600,800))

function plot_res_contour()

    ax = Axis(fig[1, 1], xlabel="petal-length", ylabel="petal-width", title="2d2class-contour")
    contour!(ax, tx, ty, probs_res; labels=true)
    colors = [:red, :blue]
    for i in 1:2
        scatter!(ax, gdf[i][:, 3], gdf[i][:, 4], color=(colors[i], 0.8), marker=:circle, markersize=10, strokewidth=1, strokecolor=:black, label=gdf[i][1, 5] == 1 ? "virginica" : "non-virginica")
    end
    axislegend(ax, position=:lt)
    #save("./imgs/iris-logreg-2d-2class-contourf.png",fig)
    fig
end

function plot_res_contourf()

    ax = Axis(fig[1, 2], xlabel="petal-length", ylabel="petal-width", title="2d2class-contourf")
    contourf!(ax, tx, ty, probs_res; levels=6, colormap=(:heat, 0.5))
    #contourf!(ax,tx,ty,yhat;levels=length(cats),colormap=(:heat,0.5))
    colors = [:red, :blue]
    for i in 1:2
        scatter!(ax, gdf[i][:, 3], gdf[i][:, 4], color=(colors[i], 0.8), marker=:circle, markersize=10, strokewidth=1, strokecolor=:black, label=gdf[i][1, 5] == 1.0 ? "virginica" : "non-virginica")
    end
    axislegend(ax, position=:lt)
    #save("./imgs/iris-logreg-2d-2class-contourf.png",fig)
    fig
end

plot_res_contourf()
plot_res_contour()

#save("./imgs/iris-logreg-2d-2class-2.png",fig)





