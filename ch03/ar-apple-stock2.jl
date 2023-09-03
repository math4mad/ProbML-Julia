"""
apple stock 股票自回归方法
参考 hankan 的 ar.jl 方法.
要注意两点:
1. 数据需要做处理. 需要做压缩,
2. 先验的参数需要做一定的调整. 
"""



using Turing, Distributions, Plots, DataFrames, Random, CSV, StatsPlots
Random.seed!(12345)





@model function autoreg(data)
    if data === missing
        # Initialize `x` if missing
        data = Vector{Float64}(undef, 240)
    end

    alpha ~ Normal(-1, 6)
    beta ~ Normal(-1, 6)
    sigma ~ Truncated(Gamma(1, 1), 0, Inf)


    for i in eachindex(data)

        if i == 1
            data[i] ~ Normal(alpha + beta, sigma)
        else
            data[i] ~ Normal(alpha + beta * data[i-1], sigma)
        end
    end

end

url = "/Users/lunarcheung/Public/DataSets/2014_apple_stock.csv"

df = CSV.File(url) |> DataFrame

data = df[:, 2]
ts = range(0, 1, 240)

function average(data)

    ys_max = maximum(data)
    yf = data .- ys_max
    return yf
end

newdata = average(data) / 30

function prior_sample()
    model = autoreg(missing)  #不传数据的采样
    @time chns =  sample(model, HMC(0.01, 5), 500)
    pys = group(chns, :data)
    df = DataFrame(pys)
    data = select(df, 3:102) |> Matrix |> d -> sample(d, (5, 240))
    scatter(ts,newdata,label=false,ms=2,alpha=0.5)
    plot!(ts, data'; lw=0.5, label=false, alpha=0.5)
    
end

function bayes_sample(data)
    model = autoreg(data)
    chns = sample(model, HMC(0.01, 5), 1000)

    df = DataFrame(chns)
    α = (select(df, 3)) |> Matrix |> mean
    β = (select(df, 4)) |> Matrix |> mean
    σ = (select(df, 5)) |> Matrix |> mean
    return α, β, σ
end

function pred_y(ts)
    α, β, σ = bayes_sample(newdata)

    predy = []

    for (idx, t) in enumerate(ts)
        if idx == 1

            push!(predy, rand(Normal(α + β, σ)))

        else
            push!(predy, rand(Normal(α + β * predy[idx-1], σ)))

        end
    end

    return predy

end


scatter(ts,newdata,label=false,ms=2,alpha=0.5)
plot!(ts, pred_y(ts); lw=0.5, label=false, alpha=0.5)



