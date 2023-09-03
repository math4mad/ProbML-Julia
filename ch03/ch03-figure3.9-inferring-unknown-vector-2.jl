
"""
probml page 119  利用数据点和先验数据得到后验分布数据

没有套用书中公式, 直接使用 Turing.jl的模型来推断后验均数

数据点的协方差矩阵已知(为了简化过程),  利用数据点和先验二元高斯分布推断后验均数向量

 Turing 贝叶斯模型 直接定义 均值先验分布来自于均值为 0, 协方差为已知的多元分布
 `μ ~ MvNormal(z2, Σy2)`

```
@model function gaussian_model(x)

    Σy1 = 0.1 * [2 1; 1 1]
    z2 = [0.0, 0.0]
    Σy2 = 0.1 * [1 0; 0 1]
    μ ~ MvNormal(z2, Σy2)
    D, N = size(x)

    for i in 1:N

        x[:, i] ~ MvNormal(μ, Σy1)
    end
end
```

"""

using GLMakie, Distributions, Random, Turing
Random.seed!(3)

z1 = [0.5, 0.5]
Σy1 = 0.1 * [2 1; 1 1]
d1 = MvNormal(z1, Σy1)


z2 = [0.0, 0.0]
Σy2 = 0.1 * [1 0; 0 1]
d2 = MvNormal(z2, Σy2)  #标准正态分布? 

#= 根据 probml page 120 的描述,先验分布的协方差是已知的, 这是已知信息
   所以不是 Σy2 = 0.1 * [1 0; 0 1]
   而是  Σy3 = 0.1 * [2 1; 1 1], 与生成数据的协方差一样
   书里这个地方的图, 我觉得有点问题
=#
z3 = [0.0, 0.0]
Σy3 = 0.1 * [2 1; 1 1]
d3 = MvNormal(z2, Σy3)  # 先验分布


fig = Figure(resolution=(1200, 400))
xs = ys = range(-1, 1, 100)


"""
    sample_data(;dist=d1,n=10)
#观察数据抽样
## 参数
1. dist 为分布
2. n为抽样数
TBW
"""
function sampling_data(;dist=d1,n=10)
    X = rand(dist, n)  #观测数据抽样
    return X
end 

#绘制观测数据

function plot_p1(X)
    ax = Axis(fig[1, 1]; limits=(-1, 1, -1, 1), title="data")
    scatter!(ax, X, size=14; label="(a)")
    scatter!(ax, [0.5], [0.5]; marker=:xcross, markersize=24)
end

#绘制先验分布
function plot_p2( xs,ys;dist=d3)
    zs = [pdf(dist, [x, y]) for x in xs, y in ys]
    ax = Axis(fig[1, 2], limits=(-1, 1, -1, 1), title="prior")
    contour!(ax, xs, ys, zs; label="(b)")
end

## bayesian posterior  infer

function bayesian_infer(X)
    x = X
    @model function gaussian_model(x)

        Σy1 = 0.1 * [2 1; 1 1]
        z2 = [0.0, 0.0]
        Σy2 = 0.1 * [1 0; 0 1]
        μ ~ MvNormal(z2, Σy2)
        D, N = size(x)

        for i in 1:N

            x[:, i] ~ MvNormal(μ, Σy1)
        end


    end

    model = gaussian_model(X)

    sampler = NUTS()

    chain = sample(model, sampler, 1_000; progress=true)
    #Plots.plot(chain)
    μ_mean = [mean(chain, "μ[$i]") for i in 1:2]

    return μ_mean
end

#返回后验参数均值向量
#z1 = bayesian_post(X)

function plot_p3(z)
    z3 = z
    Σy1 = 0.1 * [2 1; 1 1]
    d = MvNormal(z3, Σy1)
    zs = [pdf(d, [x, y]) for x in xs, y in ys]
    ax = Axis(fig[1, 3], limits=(-1, 1, -1, 1), title="post after 10 observation")
    contour!(ax, xs, ys, zs; label="(c)")
    scatter!(ax, X, size=14; label="(a)")
    scatter!(ax, [0.5], [0.5]; marker=:xcross, markersize=24)
end

X=sampling_data(;dist=d1,n=10)  #数据抽样
plot_p1(X)                      #数据绘制

#plot_p2(xs,ys;dist=d2)
plot_p2(xs,ys;dist=d3)          #先验分布

bayesian_infer(X)|>plot_p3      #后验分布和数据

fig
#save("./inferring-unknown-vector-figure3.9-2.png",fig)










