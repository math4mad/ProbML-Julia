"""
mml-book page 295  
"""

using MLJ, DataFrames, GLMakie,Distributions,Random
using StatsFuns: logistic

Random.seed!(3457);

fontsize_theme = Theme(fontsize=16)
set_theme!(fontsize_theme)
fig=Figure()
ax = Axis(fig[1, 1])

μ=0
σ=0.1
xs=range(-4,4,100)
xs2=range(-4,4,60)
xs3=range(-0.2,0.2,100)

let
    d=Normal(μ,σ)
    f(x)=sin(x)+rand(d)
    si=lines!(ax,xs,sin.(xs);linewidth=1,color=:green)
    noise=scatter!(ax,xs2,f.(xs2),color=(:red,0.5), markersize=10)
end

# for  i in [-2.5,-1,0,1,2.5]
#      local μ=sin(i)
#      local d=Normal(μ,σ)
#      fun(x)=pdf(d,x)
#      local xs=xs3.+i
#      lines!(ax,xs,fun.(xs);direction=:x)
# end

fig