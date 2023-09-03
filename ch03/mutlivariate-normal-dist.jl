"""
probml page80 figure3.5
"""

using GLMakie
using Distributions
using Latexify
using FileIO
#GLMakie.activate!()
Latexify.set_default(; starred=true)


μ = [0, 0]
Σ₁ = [2 1.8; 1.8 2]
Σ₂ = [1 0; 0 3]
Σ₃ = [1 0; 0 1]
simga_dict = Dict("full" => Σ₁, "diagonal" => Σ₂, "spherical" => Σ₃)



fig = Figure(resolution=(1400, 800))


"""
    load_img(src)
 # 加载公式图片
TBW
"""
function load_img(src)
    load("./imgs/$(src).png")
end
"""

#  二元高斯分布的三列图, 显示不同的协方差矩阵
    bionormal_plot(; μ=[0, 0], Σmatrix::Dict=simga_dict, up::Int=4, n::Int=100)
## 参数
 1.  μ,Σmatrix 为矩阵均数,Σmatrix 为默认为三列
 ```
 μ = [0, 0]
 Σ₁ = [2 1.8; 1.8 2]
 Σ₂ = [1 0; 0 3]
 Σ₃ = [1 0; 0 1]
 simga_dict = Dict("full" => Σ₁, "diagonal" => Σ₂, "spherical" => Σ₃)
 ```
 2. up 为绘图区间的上界, 下界用`-up`, n 为区间遍历步长


 !!!notice  代码内部要获取 Dict 的 keys 用于 layout的布局

 返回为两行图, 第一行为 surface 图, 第二行为 contour 图


TBW
"""
function binormal_plot(; μ=[0, 0], Σmatrix::Dict=simga_dict, up::Int=4, n::Int=100)
      xs = ys = range(-up, up, n)
      simga_keys = keys(Σmatrix)
      for (idx, val) in enumerate(simga_keys)
            cormatrix = Σmatrix[val]
            #str=latexify(cormatrix,env=:align)
            img=load_img(val)
            local d = MvNormal(μ, cormatrix)
            local zs = [pdf(d, [x, y]) for x in xs, y in ys]
            local ax1 = Axis3(fig[1, idx], title="$(val)")
            local ax2=  Axis(fig[2, idx],aspect = DataAspect(),height=60)
            local ax3 = Axis(fig[3, idx];)
            hidedecorations!(ax2)
            hidespines!(ax2)
            surface!(ax1, xs, ys, zs)
            image!(ax2, rotr90(img))
            contour!(ax3, xs, ys, zs, levels=10)
      end
end

binormal_plot()

fig
#save("./imgs/bivariable-normal-dist3.png",fig)