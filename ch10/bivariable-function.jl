using MLJ, DataFrames, GLMakie,Latexify
using StatsFuns: logistic

fontsize_theme = Theme(fontsize=16)
set_theme!(fontsize_theme)

fig = Figure()

function surface_data(w)
    fun(x, y) = w' * [x^2, y^2,1]
    span = range(-10, 10, 100)
    zs = [fun(x, y) for x in span, y in span]
    return span, span, zs
end

ws = [3,0,1,-3]
R=-2
let
    for (idx, x) in enumerate(ws)
        for (idy, y) in enumerate(ws)
            w = [x, y] == [0, 0] ? vec([0, 0.5]) : vec([x, y])
            push!(w,R)
            xs, ys, zs = surface_data(w)
            ax = Axis3(fig[idx, idy], title="w=($(w[1]),$(w[2]))")
            surface!(ax, xs, ys, zs)
        end
    end
end

fig
#save("bivariable-function.png",fig)
