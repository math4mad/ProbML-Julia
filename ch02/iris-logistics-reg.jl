"""
probml page84 figure 2.13

主要是决策边界
"""

import MLJ:predict,fit!
using MLJ,DataFrames,GLMakie,CSV

iris = load_iris();

#selectrows(iris, 1:3)  |> pretty

iris = DataFrames.DataFrame(iris);

y, X = unpack(iris, ==(:target); rng=123);

X=select!(X,3:4)

byCat = iris.target
categ = unique(byCat)
colors1 = [:orange,:lightgreen,:purple]

# grid data
n1 = n2 = 200
tx = LinRange(0, 8, 200)
ty = LinRange(-1, 4, 200)
X_test = mapreduce(collect, hcat, Iterators.product(tx, ty))
X_test = MLJ.table(X_test')
function train()
    LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels
    
    model = machine(LogisticClassifier(), X,y )
    fit!(model)
    return model
end

model=main()

ŷ = MLJ.predict(model, X_test)

res=mode.(ŷ)|>d->reshape(d,200,200)

function trans(i)
     
    if i=="setosa"
       res=1
    elseif  i=="versicolor"
       res=2
       
    else
       res=3
    end
end

ypred=[trans(res[i,j]) for i in 1:200, j in 1:200]

function  add_legend(axs)
    Legend(fig[1,2], axs,"Label";width=100,height=200)
end

function desision_boundary(ax)
    axs=[]
    for (k, c) in enumerate(categ)
        indc = findall(x -> x == c, byCat)
        #@show indc
        x=scatter!(iris[:,3][indc],iris[:,4][indc];color=colors1[k],markersize=14)
        push!(axs,x)
    end
    return axs
end

fig = Figure(resolution=(800,600))
ax=Axis(fig[1,1],xlabel="Petal length",ylabel="Petal widht",title=L"Iris Logistics classfication")
contourf!(ax,tx, ty, ypred, levels=length(categ))
axs=desision_boundary(ax)
Legend(fig[1,2],[axs...],categ)
fig

#save("./iris-logistics-reg-2.png",fig)












