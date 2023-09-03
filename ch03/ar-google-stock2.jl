"""
根据 hankank 的代码, 第二个谷歌股票换用新的模型
"""



using Turing, Distributions, Plots, DataFrames, Random, CSV, StatsPlots
Random.seed!(12345)

url="/Users/lunarcheung/Public/Julia-Code/🟢JuliaProject/ProbML-Julia/data/google_stock.csv"

df=CSV.File(url)|>DataFrame|>dropmissing

data = df[:, 2]
n=105
ts = range(0, 1, n)
function average(data)

    ys_max = maximum(data)
    yf = data .- ys_max
    return yf
end
newdata = average(data) / 80



@model function autoreg(data)
    if data === missing
        # Initialize `x` if missing
        data = Vector{Float64}(undef, 105)
    end

    alpha ~ Normal(1, 6)
    beta ~ Normal(1, 6)
    sigma ~ Truncated(Gamma(1.0, 0.1), 0, Inf)


    for i in eachindex(data)

        if i == 1
            data[i] ~ Normal(alpha + beta, sigma)
        else
            data[i] ~ Normal(alpha + beta * data[i-1], sigma)
        end
    end

end

@model function autoreg2(x)

    
    α ~ Normal(1, 6)
    β ~ Normal(1, 6)
    σ ~ Truncated(Gamma(1.0, 0.1), 0, Inf)

    if  x === missing
        # Initialize `x` if missing
        x = Vector{Float64}(undef, n)
    end

    for t in 1:n
        if t == 1
            x[t] ~ Dirac(0.0)
        elseif t == 2
            x[t] ~ Dirac(1.0)
        else
            x[t] ~ Normal(α*x[t-2] + β*x[t-1],σ)
        end
    end
end


function bayes_sample(data)
    model = autoreg2(data)
    chns = sample(model, HMC(0.01, 3), 2000)

    df = DataFrame(chns)
    α = (select(df, 3)) |> Matrix |> mean
    β = (select(df, 4)) |> Matrix |> mean
    σ = (select(df, 5)) |> Matrix |> mean
    return α, β, σ
end

function bayes_sample_test(data)
    model = autoreg2(data)
    chns=sample(model,HMC(0.01,5), 2_000)
    #chns=sample(model,NUTS(0.65), 2_000)
    display(chns)
end


function pred_y(ts)
    α, β, σ = bayes_sample(newdata)

    predy = []

    for (idx, t) in enumerate(ts)
        if idx == 1

            push!(predy, mean(Normal(α + β, σ)))

        else
            push!(predy, mean(Normal(α + β * predy[idx-1], σ)))

        end
    end

    return predy

end



#scatter(ts,newdata,label=false,ms=2,alpha=0.5)
#plot!(ts, pred_y(ts); lw=0.5, label=false, alpha=0.5)

bayes_sample_test(newdata)
