"""
概率准备  
"""

using FileIO,HTTP,Images,GLMakie,Random,Distributions

url="https://gss0.baidu.com/-4o3dSag_xI4khGko9WTAnF6hhy/zhidao/wh%3D600%2C800/sign=1d583dbe9182d158bbd751b7b03a35e0/b3119313b07eca80a96982b79c2397dda1448376.jpg"

coin_image = load(download(url))

head = coin_image[:, 1:end÷2]|>rotr90
tail = coin_image[:, end÷2:end]|>rotr90

fig=Figure(resolution=(560,2100))

Random.seed!(3457);
images=rand( [head, tail], 3, 3)

a=b=c=['H','T']

trail=[(a[i+1],b[j+1],c[k+1]) for i in 0:1, j in 0:1,k in 0:1  if i==0||j==0||k==0]
@info "扔三次硬币, 至少扔出一次正面的事件空间:$(trail)"

#trail=[('H', 'H', 'H'), ('T', 'H', 'H'), ('H', 'T', 'H'), ('T', 'T', 'H'), ('H', 'H', 'T'), ('T', 'H', 'T'), ('H', 'T', 'T')]

N=length(trail)

for i in 1:N
    ga = fig[i, 1] = GridLayout()
    Box(fig[i, 1], strokecolor = :black,strokewidth=4)
    for j in 1:3
        ax=GLMakie.Axis(ga[1,j];)
        hidedecorations!(ax)
        img=trail[i][j]=='H' ? head : tail
        image!(ax,img;)
    end
end
rowgap!(fig.layout, 20)
colgap!(fig.layout, 0.05)

fig


#save("throw-3-coin-at-least-1-head-event-space.png",fig)