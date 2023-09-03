"""
精简代码
"""

using GLMakie, Distributions, Latexify
μ = [0, 0]
Σ₁ = [2 1.8; 1.8 2]
Σ₂ = [1 0; 0 3]
Σ₃ = [1 0; 0 1]
simga_dict = Dict("full" => Σ₁, "diagonal" => Σ₂, "spherical" => Σ₃)
fig = Figure(resolution=(1400, 800))

function load_img(src)
      load("./imgs/$(src).png")
end

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
