"""
from Probml:Figure 20.3
图片方法https://scikit-learn.org/stable/modules/generated
/sklearn.datasets.fetch_olivetti_faces.html

Factor Analysis

"""

import MLJ:transform,inverse_transform
using MLJ,DataFrames,GLMakie,CSV,Random,JLSO
fontsize_theme = Theme(fontsize = 24)
set_theme!(fontsize_theme)
Random.seed!(12121)
w=h=64
length=w*h
#data processing

function data_prepare(str)
    fetch(str) = str |> d -> CSV.File("./data/$str.csv") |> DataFrame
    to_ScienceType(d)=coerce(d,:label=>Multiclass)
    df = fetch(str)
    return df
end

str="scikit_fetch_olivetti_faces"
of=olivetti_faces=data_prepare(str)

of=coerce!(of,:label=>Multiclass)
label,Xtrain=  unpack(of, ==(:label), rng=123);
rows,_=size(of)

pick25=rand(1:rows,25)
imgs=Xtrain[pick25,:]
namelabels=label[pick25]


"""
    plot_25_face(imgs,fig;row=1)
绘制原始数据的图片
输入的 row 表示原始添加到最后一行   
接下来一行显示图片的标签

👨‍🚀math4mads
"""
function plot_25_face(imgs,fig;row=1)
        imgs=Matrix(imgs)
    
    for idx in 1:25
        
        local ax = Axis(fig[row+1,idx],yreversed=true)
        local ax2=Axis(fig[row+2,idx],height=50)
        img=imgs[idx,:]|>d ->reshape(d, w, h)
        image!(ax, img)
        text!(ax2,"$(namelabels[idx])")
        hidespines!(ax) 
        hidedecorations!(ax)
        hidespines!(ax2) 
        hidedecorations!(ax2)
    end
    
end


FactorAnalysis = @load FactorAnalysis pkg=MultivariateStats



"""
    produce_model(Xtr;dim=5)
训练模型

arguments

- Xtr :训练数据
- dim :缩减到的维度

无返回值, 模型存储为 jlso 格式文件,key为:pca

👨‍🚀math4mads
"""
function  produce_model(Xtr;dim=5)
     
    local  model = FactorAnalysis(maxoutdim=dim)
    local  mach = machine(model, Xtr) |> fit!
    JLSO.save("$(pwd())/ch20/models/of-fa-model-$(dim)pcs.jlso",:pca=>mach)
     
end



"批量训练 pca model"
make_fa_models=()->let
    for dim in [1,2,3]
        produce_model(Xtrain;dim=dim)
        @info "$dim done!"
    end
end

make_fa_models() 



"""
    make_reconstruct_data(imgs; dim=1)
重建维度缩减图片

调用模型,  对 pick25 图片进行变换, 然后重建

return  Xr:  feature 的数量与原始数据一致

👨‍🚀math4mads
"""
# function make_reconstruct_data(imgs; dim=1)
#     @info "$dim pca proceeding..."

#     mach = JLSO.load("$(pwd())/ch20/models/of-model-$(dim)pcs.jlso")[:pca]
#     Yte = transform(mach, imgs)     # 降维数据
#     Xr = inverse_transform(mach, Yte)  # 重建近似数据
#     return Xr
# end


"""
    plot_reconstruct_img(dim,row;n=25)
    绘制重建图片
    
TBW
"""
# function plot_reconstruct_img(dim,row;n=25)
#    local  Xr=make_reconstruct_data(imgs; dim=dim)
#     for i in 1:n
#         local img = Xr[i, :]|>Array|> d -> reshape(d, w, h)
#         local ax = Axis(fig[row, i],yreversed=true)
#         image!(ax, img)
#         hidespines!(ax)
#         hidedecorations!(ax)
        
#     end
# end






# fig=Figure(resolution=(2500,650))

# plot_25_face(imgs,fig;row=5)
# for (row, dim) in enumerate([1,2,50,100,300])
#     plot_reconstruct_img(dim,row)
#     ax=Axis(fig[row,26],width=130)
#     text!(ax,"d=$(dim)")
#     hidespines!(ax)
#     hidedecorations!(ax)
# end


# fig
#save("6-olvetti-face-pca.png",fig)