

using MLJ, DataFrames,GLMakie

x1=[0,0,1,1]
x2=[0,1,0,1]
y=[0,1,1,0]
xor=DataFrame(x1=x1,x2=x2,y=y)
gdf=groupby(xor,:y)
markers=[:utriangle,:rect]

fig=Figure()
ax=Axis(fig[1,1])

for (i,g) in  enumerate(gdf)
    scatter!(ax,gdf[i][!,:x1],gdf[i][!,:x2];marker=markers[i], markersize=24,strokewidth=1,strokecolor=:black)
end
fig

