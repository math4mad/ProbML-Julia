using MLJ, DataFrames, GLMakie
using StatsFuns: logistic

fontsize_theme = Theme(fontsize=16)
set_theme!(fontsize_theme)

fig = Figure()

function surface_data(w)
    fun(x, y) = logistic(w' * [x, y])
    span = range(-10, 10, 100)
    zs = [fun(x, y) for x in span, y in span]
    return span, span, zs
end

ws = [3, 0, -3]
let
    for (idx, x) in enumerate(ws)
        for (idy, y) in enumerate(ws)
            w = [x, y] == [0, 0] ? vec([0, 0.5]) : vec([x, y])
            xs, ys, zs = surface_data(w)
            ax = Axis3(fig[idx, idy], title="w=($(w[1]),$(w[2]))")
            surface!(ax, xs, ys, zs)
        end
    end
end

fig
#save("sigmoid-decision-boundary-with-different-weight-figure10.2.png",fig)
