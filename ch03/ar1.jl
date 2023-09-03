#=
   AR(1) in Turing.jl

   From Stan Users Guide, section 3.1 (page 51) "Autoregressive Models"
   (Though this is a port of my WebPPL model ar1.wppl)

     $yₙ~normal(α+βyₙ₋₁,σ)$

     $yₜ=c+ϕyₜ₋₁+ε$


https://dm13450.github.io/2017/06/09/Bayesian-Auto-Process.html
    添加先验模型的采样图, 后验模型预测的图
=#

using Turing, Distributions, Plots, DataFrames, Random
Random.seed!(12345)

macro memoize(expr)
    local cache = Dict()
    local res = undef
    local params = expr.args
    #@show params
    local id = hash(params)
    if haskey(cache, id) == true
        res = cache[id]
    else
        local val = esc(expr)

        push!(cache, (id => val))
        res = cache[id]
    end

    return :($res)
end



@model function ar1(ys)
    if ys === missing
        # Initialize `x` if missing
        ys = Vector{Float64}(undef, 100)
    end

    alpha ~ Normal(2, 2)
    beta ~ Normal(2, 2)
    sigma ~ Truncated(Gamma(2, 2), 0, Inf)

    for i in eachindex(ys)
        if i == 1
            ys[i] ~ Normal(alpha + beta, sigma)
        else
            ys[i] ~ Normal(alpha + beta * ys[i-1], sigma)
        end
    end

end

ys = [0.705429, 1.43062, 0.618161, 0.315107, 1.09993, 1.49022, 0.690016, 0.587519, 0.882984,
    1.0278, 0.998615, 0.878366, 1.17405, 0.532718, 0.486417, 1.13685, 1.32453, 1.3661,
    0.914368, 1.07217, 1.1929, 0.418664, 0.889512, 1.47218, 1.13471, 0.410168, 0.639765,
    0.664874, 1.12101, 1.22703, -0.0931769, 0.4275, 0.901126, 1.01896, 1.27746, 1.17844,
    0.554775, 0.325423, 0.494777, 1.05813, 1.10177, 1.11225, 1.34575, 0.527594, 0.323462,
    0.435063, 0.739342, 1.05661, 1.42723, 0.810924, 0.0114801, 0.698537, 1.13063, 1.5286,
    0.968813, 0.360574, 0.959312, 1.2296, 0.994434, 0.59919, 0.565326, 0.855878, 0.892557,
    0.831705, 1.31114, 1.26013, 0.448281, 0.807847, 0.746235, 1.19471, 1.23253, 0.724155,
    1.1464, 0.969122, 0.431289, 1.03716, 0.798294, 0.94466, 1.29938, 1.03269, 0.273438,
    0.589431, 1.2741, 1.21863, 0.845632, 0.880577, 1.26184, 0.57157, 0.684231, 0.854955,
    0.664501, 0.968114, 0.472076, 0.532901, 1.4686, 1.0264, 0.27994, 0.592303, 0.828514,
    0.625841]

#标准化处理  参见 turing.jl  时间序列分析
function average(data)

    ys_max = maximum(data)
    yf = data .- ys_max
    return yf
end

nys = average(ys)

time = range(1, 5, 100)
t_min, t_max = extrema(time)
xs = (time .- t_min) ./ (t_max - t_min)



function prior_sample()
    model = ar1(missing)
    @time chns = @memoize sample(model, HMC(0.01, 5), 500)
    #dump(sample(model, HMC(0.01, 5), 500))
    pys = group(chns, :ys)
    df = DataFrame(pys)
    data = select(df, 3:102) |> Matrix |> d -> sample(d, (5, 100))
    newdata = []
    r, _ = size(data)
    for i in r
        push!(newdata, average(data[i, :]))
    end
    plot(xs, data'; lw=0.5, label=false, alpha=0.5)
    scatter!(xs, nys, label=false, ms=2, color=:red)

end


function bayes_sample(data)
    model = ar1(data)
    @time chns = sample(model, HMC(0.01, 5), 500)

    df = DataFrame(chns)
    # α = (select(df, 3)) |> Matrix |> mean
    # β = (select(df, 4)) |> Matrix |> mean
    # σ = (select(df, 5)) |> Matrix |> mean

    # return α, β, σ
end

bayes_sample(nys)


# function pred(xs)
#     α, β, σ = bayes_sample(nys)

#     predy = []

#     for i in eachindex(xs)
#         if i == 1

#             push!(predy, rand(Normal(α + β, σ)))

#         else
#             push!(predy, rand(Normal(α + β * ys[i-1], σ)))

#         end
#     end

#     return predy

# end

# predy = pred(xs)

#plot(xs, predy; lw=0.5, label=false, alpha=0.5)
#scatter!(xs, nys, label=false, ms=2, color=:red)




#df=DataFrame(chs)

#α,β,σ=mean(df[!,3]),mean(df[!,4]),mean(df[!,5])



# @model function decomp_model( ys)
#     t = length(ys)
#     alpha ~ Normal(2, 2)
#     beta ~ Normal(2, 2)
#     sigma ~ Truncated(Gamma(2, 2), 0, Inf)

#     for i in 1:t
#         if i == 1
#             ys[i] ~ Normal(alpha + beta, sigma)
#         else
#             ys[i] ~ Normal(alpha + beta * ys[i-1], sigma)
#         end
#     end


# end



#decomp_model(ys)


# y_prior_samples = mapreduce(hcat, 1:20) do _
#     α,β,σ= rand(decomp_model(ys))


#    t=length(ys)
#    pys=[]
#    for i in 1:t
#         if i == 1
#             push!(pys,rand(Normal(α + β, σ)))
#         else
#             push!(pys, rand(Normal(α + β * pys[i-1], σ)))
#         end
#     end
#     return pys
# end




# p1=scatter(1:time,ys;label=false)
# arr=[]
# push!(arr,p1)

# for i in 1:20
#     p=plot!(1:time,y_prior_samples[i,:];linewidth=1, alpha=0.5, color=1, label=false)
#     push!(arr,p)
# end

# plot!(arr...;label=false)





