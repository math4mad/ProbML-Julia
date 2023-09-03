import MLJ:fit,predict,predict_mode
using MLJ,BetaML
X, y = @load_iris;
modelType= @load NeuralNetworkClassifier pkg = "BetaML"

layers= [BetaML.DenseLayer(4,12,f=BetaML.relu),BetaML.DenseLayer(12,12,f=BetaML.relu),BetaML.DenseLayer(12,3,f=BetaML.relu),BetaML.VectorFunctionLayer(3,f=BetaML.softmax)];

model= modelType(layers=layers,opt_alg=BetaML.ADAM())

(fitResults, cache, report) =fit(model, 0, X, y);

est_classes= predict_mode(model, fitResults, X)

accuracy(y,est_classes) 