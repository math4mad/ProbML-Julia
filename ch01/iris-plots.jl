"""
 probml page 35 fig1.3
 findall(x -> x == c, byCat)  参考 beautiful makie 代码
"""

using MLJ,DataFrames,GLMakie
    fontsize_theme = Theme(fontsize = 10)
    set_theme!(fontsize_theme)

    iris = load_iris(); 
    iris = DataFrames.DataFrame(iris);
    #y, X = unpack(iris, ==(:target); rng=123);
    byCat = iris.target
    categ = unique(byCat)
    label=["sepal_length","sepal_width","petal_length","petal_width"]
    axs=[]
    colors1 = [:orange,:lightgreen,:purple]
   

    fig = Figure(resolution=(1400,1400))
   
function plot_diag(i,j)
        
            ax = Axis(fig[i, i])
            push!(axs,ax)
            for (j, c) in enumerate(categ)
                indc = findall(x -> x == c, byCat)
                density!(ax, iris[:,i][indc]; color = (colors1[j], 0.5), label = "$(c)",
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
        #push!(axs,ax)
            for (k, c) in enumerate(categ)
                indc = findall(x -> x == c, byCat)
                #@show indc
                scatter!(ax, iris[:,i][indc],iris[:,j][indc];color=colors1[k])
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
    
    
    
     function main()
       plot_pair()
       add_xy_label()
       add_legend()
       fig
       #save("./imgs/iris-corner-plot2.png",fig)
     end
   #save("./imgs/iris-corner-plot.png",fig)

with_theme(fontsize_theme, fontsize = 25) do
    main()
end
    