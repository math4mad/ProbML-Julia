#=

    This is a port of the R2 model Grass.cs

    We observe that the grass is wet. Did it rain?

    Output from the R2 model:
    ```
    Mean: 0.699
    Variance: 0.21061
    Number of accepted samples = 790
    ```

    草地是湿的有两种原因, 天下雨, 浇水了
    如果是阴天, 那么下雨的导致结果的概率增加
    如果不是阴天, 下雨的几率下降, 浇水的几率增加

    
=#

using Turing, StatsPlots, Distributions,DataFrames



function flip(p=0.5)
    return Bernoulli(p)
end

# Did it rain?
@model function grass() 
    cloudy ~ flip(0.5) 

    if cloudy
        rain ~ flip(0.8)
        sprinkler ~ flip(0.1)
    else
        rain ~ flip(0.2)
        sprinkler ~ flip(0.5)
    end

    temp1 ~ flip(0.7)

    wetRoof ~ Dirac(temp1 && rain)

    temp2 ~ flip(0.9)
    temp3 ~ flip(0.9)

    wetGrass ~ Dirac(temp2 && rain || temp3 && sprinkler)

    # We observe that the grass is wet.
    # Did it rain?
    true ~ Dirac(wetGrass)

end 

model = grass()


chns = sample(model, PG(5), 2_000)



df=DataFrame(chns)

rain=df[:,4]

[:mean=>mean(rain),:var=>var(rain),:std=>std(rain)]
