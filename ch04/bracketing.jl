
using GLMakie

fig=Figure(resolution=(1200,300))
f = x->sin(x)
N=4
s = 0.5
k = 2.0

 a = 7.0
 b = a + s
ya = f(a)
 yb = f(b)
xspan = collect(range(3π/2-2, stop=3π/2+5.5, length=101))




for i in 1:N
    global a,b,s,ya,yb,yc
    ax = Axis(fig[1, i];xticks = ([a,b], ["a", "b"]))
    lines!(ax, xspan,f.(xspan),color=:black)
    scatter!(ax,[a,b],[ya,yb];markersize=14,color=:red)


    if yb > ya
        a, b = b, a
        ya, yb = yb, ya
        s = -s
    end

    c, yc = b + s, f(b + s)
    if yc > yb
        a, b = a, c
    else
        a, ya, b, yb = b, yb, c, yc
        s *= k
    end
end

fig


# function bracket_minmum(f, x=0; s=1e-2, k=2.0)
#     if yb > ya
#         a, b = b, a
#         ya, yb = yb, ya
#         s = -s
#     end

#     c, yc = b + s, f(b + s)
#     if yc > yb
#         a, b = a, c
#     else
#         a, ya, b, yb = b, yb, c, yc
#         s *= k
#     end
# end