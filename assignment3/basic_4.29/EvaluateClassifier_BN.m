function P = EvaluateClassifier_BN(X, HyperParams, varargin)

W = HyperParams.W;
b = HyperParams.b;
k = numel(W);
H{1} = X;

for i = 1:k-1
    if HyperParams.use_bn
        %This is to implement the forward pass pf BN
        gamma = HyperParams.gamma{i};
        beta = HyperParams.beta{i};
        s = W{i} * H{i} + b{i};
        %calculate mean and variance of score s
        if nargin == 4
            mu = varargin{1}(i);
            v = varargin{2}(i);
        else
            mu = mean(s, 2);
            v = var(s, 0, 2) * (size(s, 2) - 1)/size(s, 2);
        end
        %Apply batch normalizetion in hints
        s_hat = BatchNormalize(s, mu, v);
        s_tilde = gamma .* s_hat + beta;
        H{i + 1} = max(0, s_tilde);
    else
        s = W{i} * H{i} + b{i};
        H{i + 1} = max(0, s);
    end
end
P = softmax(W{k} * H{k} + b{k});
%W_1 = W{1};
%W_2 = W{2};
%b_1 = b{1};
%b_2 = b{2};
%s_1 = W_1 * X + b_1;
%H = max(s_1, 0);
%s_2 = W_2 * H + b_2;
%P_2 = softmax(s_2);

end