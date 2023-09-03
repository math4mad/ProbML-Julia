"""
probml page 603 figure17-8
"""

using KernelFunctions,GLMakie,Random,LinearAlgebra,Distributions
using GaussianProcesses,Optim
Random.seed!(2343)


a=(1.0,1.0,0.1)
b =(3.0, 1.16, 0.89)

mZero1 = MeanZero()                   
kern1= SE(a[1],a[2])
logObsNoise1= a[3]  
mZero2 = MeanZero()                   
kern2= SE(b[1],b[2])
logObsNoise2= b[3] 
               


# make data
dist=Uniform(-10,10)
dist2=Uniform(-3,4)
X=rand(dist,20)|>sort
Y=[sin(X[i])+rand() for i in eachindex(X)]
gp = GP(X,Y,mZero1,kern1,logObsNoise1)
gp2 = GP(X,Y,mZero2,kern2,logObsNoise2)
optimize!(gp)
optimize!(gp2)
xs=range(-10,10,50)
#samples = rand(gp, xs, 5)|>transpose


(μ1, σ²1)=predict_y(gp,xs);
(μ2, σ²2)=predict_y(gp2,xs);

fig=Figure(resolution=(1200,600))
ax=Axis(fig[1,1],title=L"hyperparameters (l,σf,σy)=(1.0,1.0,0.1)",titlealign = :left)
ax2=Axis(fig[2,1],title=L"hyperparameters (l,σf,σy)=(3.0,1.16,0.89)",titlealign = :left)
lines!(ax,xs,μ1)
scatter!(ax,X,Y,marker='x',markersize=24)
band!(ax,xs,μ1+2*sqrt.(σ²1),μ1-2*sqrt.(σ²1),color=(:lightblue,0.5))
lines!(ax2,xs,μ2)
scatter!(ax2,X,Y,marker='x',markersize=24)
band!(ax2,xs,μ2+2*sqrt.(σ²2),μ2-2*sqrt.(σ²2),color=(:lightblue,0.5))
fig
save("./imgs/figure17-8.png",fig)

