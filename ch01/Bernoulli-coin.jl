"""
10次硬币试验 7 次成功的伯努利实验
"""

using Distributions, GLMakie

dist(p) = Bernoulli(p)
success(d) = succprob(d)
failure(d) = failprob(d)


"""
    pmf(; p=0.5, n=10, s=7)
    p 为 伯努利实验成功的参数
    n 为实验次数
    s 为成功次数
TBW
"""
function pmf(; p=0.5, n=10, s=7)
    d = dist(p)
    return ((success(d))^s) * ((failure(d))^(n - s)) * 100
end

probrange = range(0.1, 1.0, 100)
params = 0:100
n=100
s=60
fig = Figure()
ax1 = Axis(fig[1, 1])
ax2=Axis(fig[1, 2])

# for i in params
#     local row = div(i, 3)
#     local col = rem(i, 3)
#     local ax = Axis(fig[row+1, col])
#     local data = Float64[pmf(; p=j, n=n, s=s) for j in probrange]
#     lines!(ax, probrange, data, label="head=$(i)")
#     axislegend()
# end
data1 = Float64[pmf(; p=j, n=10, s=6) for j in probrange]
data2 = Float64[pmf(; p=j, n=100, s=60) for j in probrange]

lines!(ax1, probrange, data1, label="n=10;head=6")
axislegend(ax1)
lines!(ax2, probrange, data2, label="n=100;head=60")
axislegend(ax2)
fig
#save("Bernoulli-coin.png",fig)




