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
for i = 1:k - 1
    Wstar{i} = HyperParams.W{i} - eta * Grads.W{i};
    bstar{i} = HyperParams.b{i} - eta * Grads.b{i};
    gamma{i} = HyperParams.gamma{i} - eta * Grads.gamma{i};
    beta{i} = HyperParams.beta{i} - eta * Grads.beta{i};
end
Wstar{k} = HyperParams.W{k} - eta * Grads.W{k};
bstar{k} = HyperParams.b{k} - eta * Grads.b{k};

HyperParams.W = Wstar;
HyperParams.b = bstar;
HyperParams.gamma = gamma;
HyperParams.beta = beta;
end