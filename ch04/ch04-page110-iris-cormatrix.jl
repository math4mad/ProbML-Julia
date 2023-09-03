using MLJ, DataFrames, GLMakie, StatsBase

iris = load_iris();
iris = DataFrames.DataFrame(iris)
label = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
data = select(iris, 1:4) |> Matrix
iris_cov = (cov(data)) .|> d -> round(d, digits=3)
iris_cor = (cor(data)) .|> d -> round(d, digits=3)



fig = Figure(resolution=(1200, 600))
ax1 = Axis(fig[1, 1]; xticks=(1:4, label), yticks=(1:4, label), title="iris cov matrix")
ax3 = Axis(fig[1, 3], xticks=(1:4, label), yticks=(1:4, label), title="iris cor matrix")

hm = heatmap!(ax1, iris_cov)
Colorbar(fig[1, 2], hm)
[text!(ax1, x, y; text=string(iris_cov[x, y]), color=:white, fontsize=18, align=(:center, :center)) for x in 1:4, y in 1:4]

hm2 = heatmap!(ax3, iris_cor)
Colorbar(fig[1, 4], hm2)
[text!(ax3, x, y; text=string(iris_cor[x, y]), color=:white, fontsize=18, align=(:center, :center)) for x in 1:4, y in 1:4]
fig
#save("ch04-page110-iris-cov-vor-matrix-heatmap.png",fig)
