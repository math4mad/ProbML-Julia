"""
iris  2class 2 dimension  logistic classification
from  turing.jl  defaults demo
"""


using MLJ, DataFrames, GLMakie, Turing
using StatsFuns: logistic
using MLDataUtils: shuffleobs, stratifiedobs, rescale!
using Random
Random.seed!(0);
fontsize_theme = Theme(fontsize=16)
set_theme!(fontsize_theme)

iris = load_iris();
data = iris = DataFrames.DataFrame(iris);
label = ["petal_length", "petal_width"]

data[!, :target] = [r.target == "virginica" ? 1 : 0 for r in eachrow(data)]

select!(data, ([:petal_length, :petal_width, :target]))
#rescale feature
features = ["petal_length", "petal_width"]
for feature in features
    rescale!(data[!, feature]; obsdim=1)
end

byCat = data.target
categ = unique(byCat)
colors = [:red, :blue]

#first(data,10)
X = Matrix(data[:, features])
y = data[:, :target]

@model function logistic_regression(x, y, n, σ)
    intercept ~ Normal(0, σ)

    length ~ Normal(0, σ)
    width ~ Normal(0, σ)


    for i in 1:n
        v = logistic(intercept + length * x[i, 1] + width * x[i, 2])
        y[i] ~ Bernoulli(v)
    end
end;

n, _ = size(data)
m = logistic_regression(X, y, n, 1)
chain = sample(m, NUTS(), MCMCThreads(), 1_500, 3)

intercept = mean(chain[:intercept])
length = mean(chain[:length])
width = mean(chain[:width])

fig = Figure()
ax1 = Axis(fig[1, 1], xlabel="petal_length", ylabel="petal_width")


ran = range(-2.5, 2.5, 200)  # 


function pred(x, y; threshold=0.2)
    num = logistic(intercept .+ length * x + width * y)
    return num >= threshold ? 1.0 : 0.0

end

xs = [x for x in ran, y in ran]
ys = [y for x in ran, y in ran]
zs = [pred(x, y, threshold=0.25) for x in ran, y in ran]
con1 = contourf!(ax1, ran, ran, zs; levels=2, colormap=[:lightblue, :white])

sc = Vector(undef, 2)

for (j, c) in enumerate(categ)
    indc = findall(x -> x == c, byCat)
    sc[j] = scatter!(ax1, data[:, 1][indc], data[:, 2][indc], data[:, 3][indc])
end

Legend(fig[1, 2],
    sc,
    ["Iris-non-Virginica", "Iris-Virginica",])
fig
#save("iris-bayes-logistics-reg-two-variable.png",fig)


















