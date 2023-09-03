"""
 probml page42 figure1.6  room tempature on different place 
"""

using MLJ,DataFrames,GLMakie,MAT

train_data = matread("../data/moteData.mat")
df=DataFrame("x"=>train_data["X"][:,1],"y"=>train_data["X"][:,2],"tempature"=>train_data["y"][:,1])

fig = Figure(resolution=(800,800))
ax=Axis3(fig[1,1],xlabel=L"x",ylabel=L"y",zlabel=L"tempature(℃)",zlabelvisible=true)

scatter!(ax,df[!,"x"],df[!,"y"],df[!,"tempature"],color=df[!,"tempature"],markersize=14,colormap = :plasma)
save("probml-page43.png",fig)





