"""
Figure 10.10
"""

import MLJ:predict_mode,fit!,predict

using MLJ, GLMakie,Random,DataFrames,ColorSchemes,Distributions,LinearAlgebra,Turing

X, y = @load_iris

# outlierX=[4.2, 4.5, 4.0, 4.3, 4.2, 4.4]
# outliery=[1.03,1.1,0.99,1.1,1.0,1.1]|>d-> coerce(d,Multiclass)
# y=[y[i]=="virginica" ? 1 : 0 for i in eachindex(y)]|>d-> coerce(d,Multiclass)
# X=vcat(X,outlierX)|>d->reshape(d,156,1)
# y=vcat(y,outliery)

#BernoulliNBClassifier = @load BernoulliNBClassifier pkg=MLJScikitLearnInterface
#model = BernoulliNBClassifier()
#mach=machine(model, X, y)|>fit!
#predict(mach, X)

X=DataFrame(X)
y=filter( x->x in ["setosa", "versicolor"], y)




