


using DataFrames,Distributions,GLMakie,CSV,StatsBase

data="""
X1,X2,X3,X4,Y1,Y2,Y3,Y4
10,10,10,8,8.04,9.14,7.46,6.58
8,8,8,8,6.95,8.14,6.77,5.76
13,13,13,8,7.58,8.74,12.74,7.71
9,9,9,8,8.81,8.77,7.11,8.84
11,11,11,8,8.33,9.26,7.81,8.47
14,14,14,8,9.96,8.1,8.84,7.04
6,6,6,8,7.24,6.13,6.08,5.25
4,4,4,19,4.26,3.1,5.39,12.5
12,12,12,8,10.84,9.13,8.15,5.56
7,7,7,8,4.82,7.26,6.42,7.91
5,5,5,8,5.68,4.74,5.73,6.89
"""

df = CSV.File(IOBuffer(data))|>DataFrame

xs,ys=df[:,1:4],df[:,5:8]
colors=[:green, :orange,:red,:blue]
fig=Figure(resolution=(1600,400))

for i in 1:4
    mx,my=mean(xs[:,i])|>d->round(d,digits=2),mean(ys[:,i])|>d->round(d,digits=2)
    varx,vary=var(xs[:,i])|>d->round(d,digits=2),var(ys[:,i])|>d->round(d,digits=2)

    local ax=Axis(fig[1,i],title="dataset:$i",subtitle=L"\mathbb{Ex}=%$mx;\mathbb{Ey}=%$my;
    \mathbb{Vx}=%$varx;\mathbb{Vy}=%$vary;
    ",subtitlecolor=:red)
    scatter!(ax,xs[:,i],ys[:,i];marker=:circle,markersize=10,color=(colors[i],0.8),strokewidth=1,strokecolor=:black)
end

#save("ansombe-dataset.png",fig)