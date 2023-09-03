"""
add  a  table demo
"""


    using BrowseTables, Tables,DataFrames,CSV,MLJ
    # make example table, but any table that supports Tables.jl will work
    # table = Tables.columntable(collect(i == 5 ? (a = missing, b = "string", c = nothing) :
    #                                    (a = i, b = Float64(i), c = 'a'-1+i) for i in 1:10))
    #open_html_table(table) # open in browser
    #HTMLTable(table) # show HTML table using Julia's display system
   
    fetch(str)=str|>d->CSV.File(d,missingstring="NA")|>DataFrame|>dropmissing

table=fetch("./data/penguins.csv")|>d->coerce(d, :bill_length_mm =>Continuous, :bill_depth_mm => Continuous,:flipper_length_mm=>Continuous,:body_mass_g=>Continuous,:species=>Multiclass)


open_html_table(table) # open in browser
HTMLTable(table)