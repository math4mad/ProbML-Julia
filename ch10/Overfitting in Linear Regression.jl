"""
mmlbook page 305

methods from  kernelfunctions.jl
"""


using MLJ, DataFrames, GLMakie,Distributions,Random
using StatsFuns: logistic
fontsize_theme = Theme(fontsize=18)
set_theme!(fontsize_theme)

Random.seed!(3457);

d=Normal(0,0.2)

f(x)=-sin(x/5)+cos(x)





x_train = -5:0.5:5
x_test = -7:0.1:7

noise = rand(d, length(x_train))
y_train = f.(x_train)+noise
y_test = f.(x_test)


function linear_regression(X, y, Xstar)
    weights = (X' * X) \ (X' * y)
    return Xstar * weights
end


function featurize_poly(x; degree=1)
    return repeat(x, 1, degree + 1) .^ (0:degree)'
end


fig=Figure()
for (idx,deg) in  enumerate([0,1,3,4,6,9])
  let
    row,col=fldmod1(idx,3) #3 cols
    ax=Axis(fig[row,col],title="M=$(deg)",limits=(-5,5,-4,4))
    X = featurize_poly(x_train; degree=deg)
    Xstar = featurize_poly(x_test; degree=deg)
    y_pred = linear_regression(X, y_train, Xstar)
    lines!(ax,x_test,y_pred)
    scatter!(ax,x_train,y_train;marker=:cross)
  end
end

fig

