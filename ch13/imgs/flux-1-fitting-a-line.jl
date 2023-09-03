"""
导入 Flux 的第一个实例代码
"""

using Flux,GLMakie
using Flux: train!
using Statistics

actual(x) = 4x + 2

x_train, x_test = hcat(0:5...), hcat(6:10...)

y_train, y_test = actual.(x_train), actual.(x_test)

model = Dense(1 => 1)

predict = Dense(1 => 1)

loss(model, x, y) = mean(abs2.(model(x) .- y));

opt = Descent()

data = [(x_train, y_train)]

train!(loss, predict, data, opt)

loss(predict, x_train, y_train)

for epoch in 1:200
    train!(loss, predict, data, opt)
end

#loss(predict, x_train, y_train)


predy=predict(x_test)

fig=Figure()
ax=Axis(fig[1,1])

lines!(ax,1:5,predy[1,:];label="predy")
lines!(ax,1:5,y_test[1,:];label="y_test")
axislegend()
fig




