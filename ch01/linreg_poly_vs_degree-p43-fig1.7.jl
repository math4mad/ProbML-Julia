"""
probml page43  figure 1.7
多项式方法 from : kernelfunctions.jl doc
过拟合问题
"""


using Distributions,GLMakie,MLJ,Random
Random.seed!(33434)

# define data
f(x) = (x + 4) * (x + 1) * (x - 1) * (x - 3)
n=8
x_train = -5:0.5:5
x_test = -7:0.1:7

noise = rand(Uniform(-20, 20), length(x_train))
y_train = f.(x_train) + noise
y_test = f.(x_test)

function plot_polynomial()
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, x_train, y_train)
    scatter!(ax, x_train, y_train)
    fig
    #save("polynomial.png",fig)
end

function linear_regression(X, y, Xstar)
    weights = (X' * X) \ (X' * y)
    return Xstar * weights
end;

function featurize_poly(x; degree=1)
    return repeat(x, 1, degree + 1) .^ (0:degree)'
end

function featurized_fit_and_plot(degree)
    
    ax=Axis(fig[1,degree],title="$degree order")
    X = featurize_poly(x_train; degree=degree)
    Xstar = featurize_poly(x_test; degree=degree)
    y_pred = linear_regression(X, y_train, Xstar)
    scatter!(ax,x_train, y_train;marker=:circle,markersize=8,color=(:lightgreen,0.2),strokewidth=1,strokecolor=:black)
    lines!(x_test, y_pred)
end
#fig = Figure(resolution=(1000,200))
#[featurized_fit_and_plot(d) for d in 1:5]
#save("polynomial-fitting-with-n-degree.png",fig)
#fig


function plot_accuray()
    train_acc=[]
    test_acc=[]

    for degree in 1:n
        local X = featurize_poly(x_train; degree=degree)
        local Xstar = featurize_poly(x_test; degree=degree)
        local y_train_pred = linear_regression(X, y_train, X)
        local y_hat=linear_regression(X, y_train, Xstar)
        push!(test_acc,round(rms(y_hat,y_test),digits=3))
        push!(train_acc,round(rms(y_train_pred,y_train),digits=3))
       
    end
    fig = Figure()
    ax=Axis(fig[1,1],xlabel="order",ylabel="rms",title="polynomial-fitting-with-nth-order")
    xs=1:n
    ax.xticks=1:n
    lines!(ax,xs,[train_acc...],label="train_rms",linewidth=4)
    scatter!(ax,xs, [train_acc...];marker=:circle,markersize=12,color=(:red,0.2),strokewidth=1,strokecolor=:black)
    lines!(ax,xs,[test_acc...],label="test_rms",linewidth=4)
    scatter!(ax,xs, [test_acc...];marker=:circle,markersize=12,color=(:green,0.2),strokewidth=1,strokecolor=:black)
    axislegend(ax)
    fig
    #save("polynomial-fitting-overfit.png",fig)
    
end
plot_accuray()

