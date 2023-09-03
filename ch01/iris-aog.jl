using AlgebraOfGraphics,MLJ,DataFrames,GLMakie
set_aog_theme!()
iris = load_iris(); 
iris = DataFrames.DataFrame(iris);

label=["sepal_length","sepal_width","petal_length","petal_width"]

# plt=data(iris).mapping(:sepal_length => "sepal_length",:sepal_width =>"sepal_width ",
# :petal_length  => "petal_length ",:petal_width  => "petal_width ", marker = :target)*density(bandwidth=0.5)

# fig=draw(plt)





