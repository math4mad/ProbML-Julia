

using MLJ,DataFrames,GLMakie,TableTransforms,PairPlots

#fetch(str)=str|>d->CSV.File(d,missingstring="NA")|>DataFrame|>dropmissing

fetch(str)=str|>d->CSV.File(d,missingstring="NA")

# df=fetch("./data/penguins.csv")|>d->coerce(d,:species=>Multiclass, :island =>Multiclass,:sex=>Multiclass,Count => Continuous)

# X=select(df,3:6)
# y=select(df,1)
df=fetch("./data/penguins.csv")|>DataFrame|>table
