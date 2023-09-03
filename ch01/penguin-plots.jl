"""
按照 iris-plots 方法生成corner图
"""


using MLJ,DataFrames,GLMakie,PairPlots,CSV

fetch(str)=str|>d->CSV.File(d,missingstring="NA")|>DataFrame|>dropmissing

df=fetch("../data/penguins.csv")|>d->coerce(d, :bill_length_mm =>Continuous, :bill_depth_mm => Continuous,:flipper_length_mm=>Continuous,:body_mass_g=>Continuous,:species=>Multiclass)
X=df[:,3:6]

y=df[:,1]

label=["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]
axs=[] 
colors1 = [:orange,:lightgreen,:purple]
byCat = y

categ = unique(byCat)
#@show categ

fig = Figure(resolution=(1400,1400))

function plot_diag(i,j)
    
        ax = Axis(fig[i, i])
        push!(axs,ax)
        for (j, c) in enumerate(categ)
            indc = findall(x -> x == c, byCat)
            density!(ax, X[:,i][indc]; color = (colors1[j], 0.5), label = "$(c)",
                strokewidth = 1.25, strokecolor = colors1[j])
            
        end
    
end

"""
plot_cor(i,j)
生成非对角列的散点图
TBW
"""
function plot_cor(i,j)
    ax = Axis(fig[i, j])
    push!(axs,ax)
        for (k, c) in enumerate(categ)
            indc = findall(x -> x == c, byCat)
            #@show indc
            scatter!(ax, X[:,i][indc],X[:,j][indc];color=colors1[k])
        end
end

function plot_pair()
    [( i==j ? plot_diag(i,j) : plot_cor(i,j)) for i in 1:4,j in 1:4]
end

function add_xy_label()
    for i in 1:4
     axx=Axis(fig[4, i], xlabel =label[i],)
     axy=Axis(fig[i, 1], ylabel =label[i],)
    end
end

function  add_legend()
    Legend(fig[2:3, 5], axs[1],"Label";width=100,height=200)
end

function main_plot()
    plot_pair()
    add_xy_label()
    add_legend()
    return  fig
end

fig=main_plot()

#save("./imgs/penguin-corner-plot-2.png",fig)