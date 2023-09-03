"""
probml page84 figure 2.13

重新绘制决策边界图
"""

import MLJ:fit!,predict,predict_mode,predict_mean
using MLJ,DataFrames,GLMakie,CSV,Distributions
nums=100
iris = load_iris();
iris = DataFrames.DataFrame(iris);
y, X = unpack(iris, ==(:target); rng=123);

X=select!(X,3:4)

byCat = iris.target
cats = unique(byCat)
colors = [:orange,:lightgreen,:purple]
markers= [:utriangle,:circle,:diamond]


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


function eval_model()
    LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels
    mach = fit!(machine(LogisticClassifier(), X, y))
    yhat=predict_mode(mach, x_test)|>Array|>d->reshape(d,nums,nums)  
    probs=predict(mach, x_test)|>Array
    probs_res=broadcast(pdf, probs, "versicolor").|>(d->round(d,digits=2))|>d->reshape(d,nums,nums)
    return yhat,probs_res
end

yhat,probs_res=eval_model()

fig=Figure(resolution=(1600,800))

function plot_res_contour()

    ax = Axis(fig[1, 1], xlabel="petal-length", ylabel="petal-width", title="2d3class-contour")
    contour!(ax, tx, ty, probs_res; labels=true)
    
    for (idx,cat) in enumerate(cats)
        indc = findall(x -> x == cat, byCat)
        scatter!(ax,iris[:,3][indc],iris[:,4][indc];color=(colors[idx], 0.8), marker=markers[idx], markersize=10, strokewidth=1, strokecolor=:black, label="$cat")
        
    end
    axislegend(ax, position=:lt)
    #save("./imgs/iris-logreg-2d-3class-contour.png",fig)
    fig
end

function trans(i)
     if i=="setosa"
       res=1
    elseif  i=="versicolor"
       res=2
       
    else
       res=3
    end
end
ypred=[trans(yhat[i,j]) for i in 1:nums, j in 1:nums]
function plot_res_contourf()
    
    ax = Axis(fig[1, 2], xlabel="petal-length", ylabel="petal-width", title="2d3class-contourf")
    contourf!(ax, tx, ty, ypred; levels=length(cats),colormap=(:heat,0.2))
    
    for (idx,cat) in enumerate(cats)
        indc = findall(x -> x == cat, byCat)
        scatter!(ax,iris[:,3][indc],iris[:,4][indc];color=(colors[idx], 0.8), marker=markers[idx], markersize=10, strokewidth=1, strokecolor=:black, label="$cat")
        
    end
    axislegend(ax, position=:lt)
    save("./imgs/iris-logreg-2d-3class.png",fig)
    fig
end
plot_res_contour()
plot_res_contourf()