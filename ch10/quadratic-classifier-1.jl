"""
probml page 376 figure 10.7
"""

using MLJ, GLMakie,Random,DataFrames,ColorSchemes
Random.seed!(123)

X, y = make_blobs(400;centers = 5)
X=DataFrame(X)
y=vec(y)
cats=levels(y)

function boundary_data(df;n=200)
     n1=n2=n
     xlow,xhigh=extrema(df[:,:x1])
     ylow,yhigh=extrema(df[:,:x2])
     tx = range(xlow,xhigh; length=n1)
     ty = range(ylow,yhigh; length=n2)
     x_test = mapreduce(collect, hcat, Iterators.product(tx, ty));
     xtest=MLJ.table(x_test',names=[:x,:y])
     return tx,ty,xtest
 end
 
tx,ty,xtest=boundary_data(X)

fig=Figure()
ax=Axis(fig[1,1])

cbarPal = :thermal
cmap = cgrad(colorschemes[cbarPal], length(cat), categorical = true)
for (idx, c) in enumerate(cats)
     local data=X[y.==c,:]
     scatter!(ax, data[:,:x1],data[:,:x2];color=(cmap[idx],0.6),strokecolor = :black, strokewidth = 1)
end
#fig





