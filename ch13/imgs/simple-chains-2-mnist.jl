using CSV,DataFrames,SimpleChains,GLMakie
train_url="/Users/lunarcheung/Public/DataSets/mnist_train.csv"
test_url="/Users/lunarcheung/Public/DataSets/mnist_test.csv"

fetch(str)=str|>d->CSV.File(d,missingstring="NA")|>DataFrame|>dropmissing

trdf=train_df=fetch(train_url)

train_y=trdf[:,1]
train_y1=UInt32.(train_y .+ 1);
train_x=trdf[:,2:end]|>Matrix|>transpose|>d->reshape(d,28,28,1,:)

#test data
testdf=fetch(test_url)
test_y=testdf[:,1]
test_y1=UInt32.(test_y .+ 1);
test_x=testdf[:,2:end]|>Matrix|>transpose|>d->reshape(d,28,28,1,:)


#=====workflow==============================================================#
lenet = SimpleChain(
  (static(28), static(28), static(1)),
  SimpleChains.Conv(SimpleChains.relu, (5, 5), 6),
  SimpleChains.MaxPool(2, 2),
  SimpleChains.Conv(SimpleChains.relu, (5, 5), 16),
  SimpleChains.MaxPool(2, 2),
  Flatten(3),
  TurboDense(SimpleChains.relu, 120),
  TurboDense(SimpleChains.relu, 84),
  TurboDense(identity, 10),
)

lenetloss = SimpleChains.add_loss(lenet, LogitCrossEntropyLoss(train_y1));
#========end=============================================================#


@time p = SimpleChains.init_params(lenet);


@time p = SimpleChains.init_params(lenet, size(train_x));


estimated_num_cores = (Sys.CPU_THREADS ÷ ((Sys.ARCH === :x86_64) + 1));
G = SimpleChains.alloc_threaded_grad(lenetloss);

@time SimpleChains.train_batched!(G, p, lenetloss, train_x, SimpleChains.ADAM(3e-4), 10);
SimpleChains.accuracy_and_loss(lenetloss, train_x, p)
SimpleChains.accuracy_and_loss(lenetloss, test_x, test_y, p)








