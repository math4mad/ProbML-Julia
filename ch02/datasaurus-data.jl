"""
datasaurus
probml page73, fig2.6

add  ColorSchemes  method 
"""

using DataFrames,CSV,GLMakie,ColorSchemes

(gdf,len)=let
    path="../data/datasaurus.csv"
    fetch(str)=CSV.File(str)|>DataFrame
    df=fetch(path)
    gdf = groupby(df, :dataset) 
    len=length(keys(gdf))
    gdf,len
end

cbarPal = :thermal
cmap = cgrad(colorschemes[cbarPal], len, categorical = true)

function plot_datasaurus(gdf,len)
    fig=Figure(resolution=(1200,1200))
    for i in 1:len
        local row,col=fldmod1(i,4)
        local ax=Axis(fig[row,col],title="dataset:$(gdf[i][1,1])")
        local xs,ys=gdf[i][:,2],gdf[i][:,3]
        scatter!(ax,xs,ys;marker=:circle, color=(cmap[i],0.6),markersize=8,strokewidth=1,strokecolor=:black)
    end
    save("./imgs/datasaurus-colorschemes.png",fig)
    fig
end

plot_datasaurus(gdf,len)
