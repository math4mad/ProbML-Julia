using SimpleChains,GLMakie

model = SimpleChain(
  static(1),
  TurboDense(identity,1),
  
)

actual(x) = 4x + 2

x_train, x_test = hcat(0:5...), hcat(6:10...)

y_train, y_test = actual.(x_train), actual.(x_test)

@time p = SimpleChains.init_params(model);

G = SimpleChains.alloc_threaded_grad(model);

modelloss = SimpleChains.add_loss(model, SquaredLoss(y_train));
modeltest = SimpleChains.add_loss(model, SquaredLoss(y_test));


report = let mtrain = modelloss, X=x_train, Xtest=x_test, mtest = modeltest
    p -> begin
      let train = modelloss(X, p), test = modeltest(Xtest, p)
        @info "Loss:" train test
      end
    end
  end

report(p)


for _ in 1:3
    @time SimpleChains.train_unbatched!(
      G, p, modelloss, x_train, SimpleChains.ADAM(), 10_000
    );
    report(p)
  end

  