"""
probml page 635 figure18.4
"""

import MLJ:predict_mode,fit!,machine
using MLJ,GLMakie,Random,DataFrames

nums=100
df,gdf=(let 
    X, y = make_moons(200; noise=0.1)
    df = DataFrame(X)
    df.y=y
    gdf=groupby(df,:y)
    (df,gdf)
end)

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
tx,ty,x_test=boundary_data(df)

X=df[!,1:2]
y=df[!,end]

#define model
models,titles=(let
    Tree = @load DecisionTreeRegressor pkg=DecisionTree
    RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree
    singleTree=Tree()
    forest10 = EnsembleModel(model=singleTree;bagging_fraction=0.1)
    forest50 = EnsembleModel(model=singleTree;bagging_fraction=0.5)
    randomForest = RandomForestClassifier(;sampling_fraction=0.5)
    models=[singleTree,forest10,forest50,randomForest]
    titles=["single tree","bagging tree n=10","bagging tree n=50","random forest n=50"]
    (models,titles)
end)

# plot boundary 
(let fig=Figure()
    for i in 1:4
        let
            row,col=fldmod1(i,2)
            ax=Axis(fig[row,col],title="$(titles[i])")
            mach = machine(models[i],X,y) |> fit!
            yhat=predict_mode(mach, x_test)|>Array|>d->reshape(d,nums,nums)
            contourf!(ax,tx,ty,yhat,levels=2,colormap=:heat)
            scatter!(ax,gdf[1][!,:x1],gdf[1][!,:x2])
            scatter!(ax,gdf[2][!,:x1],gdf[2][!,:x2])
        end
    end
fig ;#save("./imgs/figure18.4.png",fig) 
end)

