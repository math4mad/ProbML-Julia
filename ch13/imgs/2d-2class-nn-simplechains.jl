"""
page455 13.2.4.1
"""


using SimpleChains,GLMakie,MLJ
iris = load_iris();
iris = DataFrames.DataFrame(iris);
y, X = unpack(iris, ==(:target); rng=123);
function trans(i)
  if i=="setosa"
    res=1
 elseif  i=="versicolor"
    res=2
    
 else
    res=3
 end
end
y=[trans(i) for i in y]
(Xtrain, Xtest), (ytrain, ytest)  = partition((X, y), 0.7, multi=true,  rng=123)


    mlpd = SimpleChain(
    static(4),
    TurboDense(tanh, 3)
    
  )


  @time p = SimpleChains.init_params(mlpd);
  G = SimpleChains.alloc_threaded_grad(mlpd);

  mlpdloss = SimpleChains.add_loss(mlpd, SquaredLoss(ytrain));
  mlpdtest = SimpleChains.add_loss(mlpd, SquaredLoss(ytest));

  report = let mtrain = mlpdloss, X=Xtrain, Xtest=Xtest, mtest = mlpdtest
      p -> begin
        let train = mtrain(X, p), test = mtest(Xtest, p)
          @info "Loss:" train test
        end
      end
  end
    
  report(p)
    









