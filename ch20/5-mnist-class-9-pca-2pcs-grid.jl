"""
from Probml:Figure 20.2

绘制 4 分位标志线, 从降维的两维数据中获取四分位数
```
    x1q=Statistics.quantile(Ytr.x1)
    x2q=Statistics.quantile(Ytr.x2)
    vlines!(ax,[x1q...],color=:red,linestyle=:dot)
    hlines!(ax,[x2q...],color=:red,linestyle=:dot)
```
"""

import MLJ:transform
using MLJ,DataFrames,GLMakie,CSV,Random,StatsBase,Statistics
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

str="mnist_train"
mnist=data_prepare(str)
totalrows,_=size(mnist)

PCA = @load PCA pkg=MultivariateStats
maxdim=2
model=PCA(maxoutdim=maxdim)

"""
    selected_digitspca(num::Int;rows=25)->let
挑选一个数字, 返回图片,整体降维数据和随机选出的 25行数字降维数据

`arguments`
- num 0:9 的数字

`return`
-  imgs 选出的 25 个数字的矩阵形式
-  Ytr, 所有选出数字的降维数据
-  selectedYtr 选出的 25行降维数据
"""
function selected_digits_pca(num::Int;rows=25)
    digs=filter(:label => ==(num),mnist)
    len,_=size(digs)
    _,Xtrain=  unpack(digs, ==(:label), rng=123);
    pick25=rand(1:len,rows)
    imgs=Xtrain[pick25,:]|>Matrix
    mach = machine(model, Xtrain) |> fit!
    Ytr =transform(mach, Xtrain)
    selectedYtr=Ytr[pick25,:]
    return imgs,Ytr,selectedYtr
end



mainTitle=(gt,num)->let
    axt=Axis(gt[1,1],title="Mnist Digits=$(num) PCA",height=0,titlesize=28)
    hidedecorations!(axt)
    hidespines!(axt)
end
leftPanel=(gl,Ytr,selectedYtr)->let
    ax=Axis(gl[1,1], xlabel="First Principal  Components"
    ,ylabel="Second Principal  Components",xlabelsize=16,ylabelsize=16)
    scatter!(ax, Ytr.x1,Ytr.x2,color=(:lightgreen, 0.3))
    scatter!(ax, selectedYtr.x1,selectedYtr.x2,color=(:red, 0.9),marker=:cross,markersize=16)
    x1q=Statistics.quantile(Ytr.x1)
    x2q=Statistics.quantile(Ytr.x2)
    vlines!(ax,[x1q...],color=:red,linestyle=:dot)
    hlines!(ax,[x2q...],color=:red,linestyle=:dot)

end

rightPanel=(gr,imgs)->let
    for i in 0:4
        for j in 1:5
            idx=i*5+j
            local ax = Axis(gr[i, j],yreversed=true)
            img=imgs[idx,:]|>d ->reshape(d, 28, 28)
            image!(ax, img)
            hidespines!(ax)
            hidedecorations!(ax)
        end

    end
end

_plotinit=()->begin
    fig=Figure(resolution=(1600,800),backgroundcolor = RGBf(0.98, 0.98, 0.98))
    gt= fig[0, 1:2] = GridLayout()
    gl = fig[1, 1] = GridLayout()
    gr = fig[1, 2] = GridLayout()
    return fig,gt,gl,gr
end

function plot_pca(num)
    imgs,Ytr,selectedYtr=selected_digits_pca(num)
    fig,gt,gl,gr=_plotinit()

    mainTitle(gt,num)
    leftPanel(gl,Ytr,selectedYtr)
    rightPanel(gr,imgs)
    #save("./ch20/imgs/mnist-pca-2pcs-grid.png",fig)
    
end

# for num in 0:9
#     plot_pca(num)
# end

plot_pca(8)





