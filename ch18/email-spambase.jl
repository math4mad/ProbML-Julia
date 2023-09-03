"""
probml page 635
"""

import MLJ:predict_mode,fit!
using MLJ,GLMakie,Random,DataFrames,CSV

url1="../data/spam-datset/train_data.csv"
url2="../data/spam-datset/test_features.csv"
df=DataFrame(CSV.File(url1))

rows,cols=size(df)
df[!, :ham] = [r.ham == 1 ? 1.0 : 0.0 for r in eachrow(df)]
toSciType(df)=coerce(df,:ham =>Multiclass,Count=>Continuous)
df=toSciType(df)


X,y=df[!,1:end-2],df[!,end-1]

(Xtrain, Xtest), (ytrain, ytest)  = partition((X, y), 0.7, multi=true,  rng=123)

tree_nums=[1,100,200,300,400,500]

RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree
randomforest(n)=RandomForestClassifier(;n_trees=n)

GradientBoostingClassifier = @load GradientBoostingClassifier pkg=MLJScikitLearnInterface
boost(n)=GradientBoostingClassifier(;n_estimators=n)

BaggingClassifier = @load BaggingClassifier pkg=MLJScikitLearnInterface
bagging(n)=BaggingClassifier(;n_estimators=n)

models=[randomforest,boost,bagging]
titles=["randomforest","boost","bagging"]
markers=[:circle,:rect,:utriangle]

function train_model(model)
    res=[]
    for n in tree_nums
        mach = machine(model(n),Xtrain,ytrain) |> fit!
        yhat=predict_mode(mach, Xtest)|>Array
        acc=accuracy(ytest,yhat)
        push!(res,1-acc)
    end
    return res
end

#res=train_model.(models)


function plot_res()
    fig=Figure()
    ax=Axis(fig[1,1])
    ax.xlabel="Number of trees"
    ax.ylabel="Test error"

    for i in eachindex(res)
        ys=res[i].|>d->round(d,digits=4)
        scatterlines!(ax,tree_nums,ys;label=L"%$(titles[i])",marker=markers[i],strokewidth = 1,markersize=20,linewidth=2)
    end
    axislegend(ax)
    #save("./imgs/email-spam.png",fig)
end