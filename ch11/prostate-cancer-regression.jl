"""
probml page 416  fig11-11
"""

using MLJ,CSV,DataFrames,GLMakie,Random
Random.seed!(3344)

str="../data/prostate.csv"

fetch(str)=CSV.File(str)|>DataFrame

df=fetch(str)
first(df,10)
