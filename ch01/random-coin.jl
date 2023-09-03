"""
概率准备
"""

using FileIO,HTTP,Images,GLMakie,Random

url="https://gss0.baidu.com/-4o3dSag_xI4khGko9WTAnF6hhy/zhidao/wh%3D600%2C800/sign=1d583dbe9182d158bbd751b7b03a35e0/b3119313b07eca80a96982b79c2397dda1448376.jpg"

coin_image = load(download(url))

head = coin_image[:, 1:end÷2]|>rotr90
tail = coin_image[:, end÷2:end]|>rotr90

fig=Figure(resolution=(900,930))

Random.seed!(3457);
images=rand( [head, tail], 3, 3)

for i in 1:9
    row,col=fldmod1(i,3)
    ax=GLMakie.Axis(fig[row,col])
    hidedecorations!(ax)
    image!(ax,images[row,col])
end

fig