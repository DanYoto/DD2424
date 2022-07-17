function HyperParams = MiniBatchGD(X, Y, HyperParams, lambda, eta, alpha)

Grads = ComputeGradients(X, Y, HyperParams, lambda);

W = HyperParams.W;
k = numel(W);
%In this part, we need to consider exponential moving average
if HyperParams.use_bn
    mu = Grads.mu;
    v = Grads.v;
    for i = 1: k - 1
        mu_av{i} = alpha * mu{i} + (1 - alpha) * mu{i};
        v_av{i} = alpha * v{i} + (1 - alpha) * v{i};
    end
    HyperParams.mu_av = mu_av;
    HyperParams.v_av = v_av;
end
for i = 1:k
    Wstar{i} = HyperParams.W{i} - eta * Grads.W{i};
    bstar{i} = HyperParams.b{i} - eta * Grads.b{i};
end

HyperParams.W = Wstar;
HyperParams.b = bstar;
end