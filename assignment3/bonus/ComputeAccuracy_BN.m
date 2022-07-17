function acc = ComputeAccuracy_BN(P, y)

[~, index] = max(P);
acc = sum(index == y)/length(P);
end