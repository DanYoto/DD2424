function HyperParams = InitialParams(K, d, m, use_bn)
%K = 10
%d = 3072

HyperParams.use_bn = use_bn;
k = size(m,2) + 1;
m = [d m K];

for i = 1:k
    tmp = sqrt(2/(m(i) + m(i + 1))); %Xavier
    HyperParams.W{i} = tmp * randn(m(i+1), m(i));
    %HyperParams.W{i} = 1e-1 * randn(m(i + 1), m(i)); % sig = 1e-1
    %HyperParams.W{i} = 1e-3 * randn(m(i + 1), m(i)); % sig = 1e-3
    %HyperParams.W{i} = 1e-4 * randn(m(i + 1), m(i)); % sig = 1e-4
    HyperParams.b{i} = zeros(m(i+1), 1);
    if use_bn
        HyperParams.gamma{i} = ones(m(i + 1), 1);
        HyperParams.beta{i} = zeros(m(i + 1), 1);
    end
end

%code in assignment 2
%W_1 = 1/sqrt(d) * randn(m, d);
%b_1 = zeros(m,1);
%W_2 = 1/sqrt(m) * randn(K, m);
%b_2 = zeros(K,1);
%W = {W_1, W_2};
%b = {b_1, b_2};
end