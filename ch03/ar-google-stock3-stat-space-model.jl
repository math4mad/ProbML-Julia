"""
直接使用状态空间模型
https://lampspuc.github.io/StateSpaceModels.jl/latest/
"""



using  Plots, DataFrames, Random, CSV, StateSpaceModels
Random.seed!(12345)

url="/Users/lunarcheung/Public/Julia-Code/🟢JuliaProject/ProbML-Julia/data/google_stock.csv"

gs=CSV.File(url)|>DataFrame|>dropmissing

function average(data)

    ys_max = maximum(data)
    yf = data .- ys_max
    return yf
end

data=log_data=log.(gs[:, 2])

plt = plot(gs.date, data, label = "Google stock price")

#model = LocalLevel(data)
model=BasicStructural(log_data, 12)
fit!(model)

# filter_output = kalman_filter(model)

# plot!(plt, gs.date, get_filtered_state(filter_output), label = "Filtered level",ls=:dot,lw=3)

# smoother_output = kalman_smoother(model)
# plot!(plt, gs.date, get_smoothed_state(smoother_output), label = "Smoothed level")
#savefig("google-stock-price-stat-space-model.png")

forec = forecast(model, 24)

plot(model, forec; legend = :topleft)