using GLMakie,Distributions,StatsFuns

f(x)=logistic(x)
xs=range(-4,4,100)
fig=Figure()
ax=Axis(fig[1,1],title="sigmoid function")
lines!(ax,xs,f.(xs))
#save("./imgs/sigmoid-function.png",fig)