
"""
using   InformationGeometry   KullbackLeibler  
"""

using InformationGeometry
using LinearAlgebra, Distributions

res=KullbackLeibler(Normal(-4.01,2.), Normal(-4.,2.04), HyperCube([-100,100]); tol=1e-12)

