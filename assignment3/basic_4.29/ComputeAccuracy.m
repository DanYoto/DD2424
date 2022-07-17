function acc = ComputeAccuracy(X, y, HyperParams)

P = EvaluateClassifier(X, HyperParams);
[~, index] = max(P);
acc = sum(index == y)/length(P);
end