using Turing, StatsPlots, DataFrames
using StatsFuns


@model function logistic_regression2(x, labels)
    m ~ Normal(0, 1)
    b ~ Normal(0, 1)

    n = length(x)
    v = tzeros(n)
    for i in 1:n
        v[i] = logistic(b + m*x[i])
        labels[i] ~ Bernoulli(v[i])
        
    end

    # Post predictive of labels
    labels_post = tzeros(n)
    for i in 1:n
        labels_post[i] ~ Bernoulli(v[i])
    end
end

xs = [-10, -5, 2, 6, 10]
labels = [0, 0, 1, 1, 1]
model = logistic_regression2(xs,labels)

# chns = sample(model, Prior(), 10_000)
# chns = sample(model, MH(), 10_000)
chns = sample(model, PG(15), 1_000)
# chns = sample(model, SMC(), 10_000)
# chns = sample(model, IS(), 10_000)

display(chns)
