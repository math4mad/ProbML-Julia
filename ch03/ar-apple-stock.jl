using Turing, Distributions, Plots, DataFrames, Random,CSV
Random.seed!(12345)





@model function autoreg(data)
    # missing 的解决办法: https://storopoli.io/Bayesian-Julia/pages/04_Turing/#fndef:visualization
    if data === missing
        # Initialize `x` if missing
        data = Vector{Float64}(undef, 240)
    end

    alpha ~ Normal(2, 2)
    beta ~ Normal(2, 2)
    sigma ~ Truncated(Gamma(2, 2), 0, Inf)
    
    
    for i in 1:eachindex(data)
        if i == 1
            @show  data[i]
            data[i] ~ Normal(alpha + beta, sigma)
        else
            data[i] ~ Normal(alpha + beta * data[i-1], sigma)
        end
    end

end

url="/Users/lunarcheung/Public/DataSets/2014_apple_stock.csv"

df=CSV.File(url,missingstring="NA")|>DataFrame

data=df[:, 2]
ts=range(0,1,240)
function average(data)

    ys_mean = mean(data)
    yf = data .- ys_mean
    return yf
end

price=(data).|>(d->round(d,digits=2))




#origin_data=plot(ts,price[:,1],label=false,ms=2,alpha=0.5)




#prior_sample()

function bayes_sample(data)
    model = autoreg(data)
    chns = sample(model, HMC(0.01, 5), 100)

    df = DataFrame(chns)
    # α = (select(df, 3)) |> Matrix |> mean
    # β = (select(df, 4)) |> Matrix |> mean
    # σ = (select(df, 5)) |> Matrix |> mean
    # return α, β, σ
end

model = autoreg(data)
chns = sample(model, HMC(0.01, 5), 500)



# function pred_y(ts)
#     α, β, σ = bayes_sample(nprice)
    
#     predy = []

#     for (idx,t) in enumerate(ts)
#         if idx == 1

#             push!(predy, rand(Normal(α + β, σ)))

#         else
#             push!(predy, rand(Normal(α + β * predy[idx-1], σ)))

#         end
#     end

#     return predy

# end

# predy =pred_y(ts)
# predy[1:20]
#npredy=(predy).|>(d->round(d,digits=3))

# scatter(ts, nprice, label=false, ms=2, color=:red)
# plot!(ts, npredy; lw=0.5, label=false, alpha=0.5)


# function prior_sample()
#     model = autoreg(missing)
#     @time chns = @memoize sample(model, HMC(0.01, 5), 500)
#     #dump(sample(model, HMC(0.01, 5), 500))
#     pys = group(chns, :ys)
#     df = DataFrame(pys)
#     data = select(df, 3:102) |> Matrix |> d -> sample(d, (5, 240))
#     newdata = []
#     r, _ = size(data)
#     for i in r
#         push!(newdata, average(data[i, :]))
#     end
#     scatter(ts,nprice,label=false,ms=2,alpha=0.5)
#     plot!(ts, data'; lw=0.5, label=false, alpha=0.5)
    

# end

