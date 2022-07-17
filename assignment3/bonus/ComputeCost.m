function [c, l] = ComputeCost(X, Y, HyperParams, lambda)

W = HyperParams.W;
b = HyperParams.b;
len = size(Y,2);
H{1} = X;
sumW = sum(sum(W{1}.^2));
k = numel(W);

for i = 1:k - 1
    s = W{i} * H{i} + b{i};
    H{i + 1} = max(s, 0);
    sumW = sumW + sum(sum(W{i+1}.^2));
end

P = softmax(W{k} * H{k} + b{k});

l = -sum(log(sum(Y .* P, 1)))/len;
c = l + lambda * sumW;


%D = size(Y,2); 
%W_1 = W{1};
%W_2 = W{2}; 
%b_1 = b{1}; 
%b_2 = b{2};
    
%H = max(W_1*X+b_1,0); 
%P = softmax(W_2*H+b_2); 
    
%l = -sum(log(sum(Y .* P, 1)))/D;
%c = l + lambda*(sum(sum(W_1.^2))+sum(sum(W_2.^2)));    

end