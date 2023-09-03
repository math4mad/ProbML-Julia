"""
probml page 166 fig-4.14
"""

using  Distributions,GLMakie,Random
Random.seed!(34343)

d1=Dirichlet(3, 1)  
