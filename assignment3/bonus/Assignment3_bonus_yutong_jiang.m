addpath '/Users/yotod/Master/DD2424 Deep learning/MATLAB version/Datasets/cifar-10-batches-mat';


%Load the data
%[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');

[X_train_1, Y_train_1, y_train_1] = LoadBatch('data_batch_1.mat');
[X_train_2, Y_train_2, y_train_2] = LoadBatch('data_batch_2.mat');
[X_train_3, Y_train_3, y_train_3] = LoadBatch('data_batch_3.mat');
[X_train_4, Y_train_4, y_train_4] = LoadBatch('data_batch_4.mat');
[X_train_5, Y_train_5, y_train_5] = LoadBatch('data_batch_5.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

X_train = [X_train_1, X_train_2, X_train_3, X_train_4, X_train_5];
Y_train = [Y_train_1, Y_train_2, Y_train_3, Y_train_4, Y_train_5];
y_train = [y_train_1, y_train_2, y_train_3, y_train_4, y_train_5];


X_val = X_train(:, 1:5000);
Y_val = Y_train(:, 1:5000);
y_val = y_train(:, 1:5000);

X_train = X_train(:, 5001:end);
Y_train = Y_train(:, 5001:end);
y_train = y_train(:, 5001:end);


%parameter setting
%for k-leyer network, input m should be a 1*(k-1) vector
lambda = 0.00358;
%lambda = 0.00319;
n_batch = 100;
n = size(X_train, 2);
%n_s = 2 * floor(n/n_batch);
n_s = 5 * 45000/n_batch;
eta_min = 1e-5;
eta_max = 1e-1;
n_cycle = 2;
n_epochs = 2 * n_s * n_cycle * n_batch / n;
t = 1;
l = 0;
alpha = 0.9;
%count = 0;

%pre-process raw data
mean_X = mean(X_train, 2);
std_X = std(X_train, 0, 2);
X_train = X_train - repmat(mean_X, [1, size(X_train, 2)]);
X_train = X_train ./ repmat(std_X, [1, size(X_train, 2)]);

X_val = X_val - repmat(mean_X, [1, size(X_val, 2)]);
X_val = X_val ./ repmat(std_X, [1, size(X_val, 2)]);

X_test = X_test - repmat(mean_X, [1, size(X_test, 2)]);
X_test = X_test ./ repmat(std_X, [1, size(X_test, 2)]);


%initializing paraneter W and d
K = size(Y_train, 1); %10
d = size(X_train, 1); %3072
%m = [50 30 20 20 10 10 10 10]; %9 layers
%skip_layer = [0 0 1 0 1 0 0];
m = [50 30 20 20 10 10];
%m = [200 100 50 50 20 20];
use_bn = 1;
HyperParams = InitialParams(K, d, m, use_bn);
k = numel(HyperParams.W); %number of layer


%check the gradients work or not
%Grads_my = ComputeGradients(X_train(:,1:10), Y_train(:,1:10), HyperParams, lambda);
%Grads_num = ComputeGradsNumSlow(X_train(:,1:10), Y_train(:,1:10), HyperParams, lambda, 1e-5);
%relative_error = max(max(abs(Grads_my.W{1} - Grads_num.W{1})));


%train_cost = zeros(2 * n_s, 1);
%val_cost = zeros(2 * n_s, 1);
%train_loss = zeros(2 * n_s, 1);
%val_loss = zeros(2 * n_s, 1);
%train_acc = zeros(2 * n_s, 1);
%val_acc = zeros(2 * n_s, 1);


%eta = 5e-3;
for epoch = 1:n_epochs
    %eta = eta * 0.8;
    for batch = 1: n/n_batch
        
        %cyclical
        if t >= 2 * l * n_s && t <= (2 * l + 1) * n_s
            eta = eta_min + (t - 2*l*n_s)*(eta_max - eta_min)/n_s;
        end
        if t >= (2*l + 1)*n_s && t <= 2*(l+1)*n_s
            eta = eta_max - (t - (2*l + 1)*n_s)*(eta_max - eta_min)/n_s;
        end
        
        j_start = (batch - 1) * n_batch + 1;
        j_end = batch * n_batch;
        X_tr_ba = X_train(:, j_start:j_end);
        Y_tr_ba = Y_train(:, j_start:j_end);
        y_tr_ba = y_train(:, j_start:j_end);
        
        HyperParams = MiniBatchGD(X_tr_ba, Y_tr_ba, HyperParams, lambda, eta, alpha);
        
        %{
        if mod(t, 50) == 1
            [c_tr, l_tr] = ComputeCost_BN(X_tr_ba, Y_tr_ba, HyperParams, lambda);
            [c_val, l_val] = ComputeCost_BN(X_val, Y_val, HyperParams, lambda);
                
            Cost_train(t) = c_tr;
            loss_train(t) = l_tr;
            Cost_val(t) = c_val;
            loss_val(t) = l_val;
        end
        %}
        t = t + 1;

        if t > 2 * (l + 1) * n_s
            l = l + 1;
        end
        
    end
    index = randperm(n);
    X_train = X_train(:, index);
    Y_train = Y_train(:, index);
end

%x = 1:t-1;
P = EvaluateClassifier_BN(X_test, HyperParams);
acc = ComputeAccuracy_BN(P, y_test);


%{
%Adam
AdamParameters = InitialAdam(HyperParams);
eta = 5e-3;

%n_epochs = 10;
%n_epochs = 80;
for i = 1:n_epochs
    
%    if mod(i,18) == 0
%        lr = lr*0.6;
%    end

    %lr = lr * 0.7
    eta = eta * 0.8;

    for batch = 1:n/n_batch
        
        j_start = (batch - 1) * n_batch + 1;
        j_end = batch * n_batch;
        X_tr_ba = X_train(:, j_start:j_end);
        Y_tr_ba = Y_train(:, j_start:j_end);
        y_tr_ba = y_train(:, j_start:j_end);
        Grads = ComputeGradients(X_tr_ba, Y_tr_ba, HyperParams, lambda);
        if HyperParams.use_bn
            for l =1:k
                AdamParameters.mW{l} = 0.9 * AdamParameters.mW{l} + 0.1 * Grads.W{l};
                AdamParameters.mb{l} = 0.9 * AdamParameters.mb{l} + 0.1 * Grads.b{l};
                AdamParameters.vW{l} = 0.999 * AdamParameters.vW{l} + 0.001 * Grads.W{l} .* Grads.W{l};
                AdamParameters.vb{l} = 0.999 * AdamParameters.vb{l} + 0.001 * Grads.b{l} .* Grads.b{l};
                mmW = AdamParameters.mW{l} / 0.1;
                mmb = AdamParameters.mb{l} / 0.1;
                vvW = AdamParameters.vW{l} / 0.001;
                vvb = AdamParameters.vb{l} / 0.001;
                HyperParams.W{l} = HyperParams.W{l} - eta./((vvW).^(1/2)+1e-8) .* mmW;
                HyperParams.b{l} = HyperParams.b{l} - eta./((vvb).^(1/2)+1e-8) .* mmb;
                if l < k
                    AdamParameters.mgammas{l} = 0.9 * AdamParameters.mgammas{l} + 0.1 * Grads.gamma{l};
                    AdamParameters.mbetas{l} = 0.9 * AdamParameters.mbetas{l} + 0.1 * Grads.beta{l};
                    AdamParameters.vgammas{l} = 0.999 * AdamParameters.vgammas{l} + 0.001 * Grads.gamma{l} .* Grads.gamma{l};
                    AdamParameters.vbetas{l} = 0.999 * AdamParameters.vbetas{l} + 0.001 * Grads.beta{l} .* Grads.beta{l};
                    mmgammas = AdamParameters.mgammas{l} / 0.1;
                    mmbetas = AdamParameters.mbetas{l} / 0.1;
                    vvgammas = AdamParameters.vgammas{l} / 0.001;
                    vvbetas = AdamParameters.vbetas{l} / 0.001;
                    HyperParams.gammas{l} = HyperParams.gamma{l} - eta./((vvgammas).^(1/2)+1e-8) .* mmgammas;
                    HyperParams.betas{l} = HyperParams.beta{l} - eta./((vvbetas).^(1/2)+1e-8) .* mmbetas;
                end
            end
        end

    end
end

P = EvaluateClassifier_BN(X_test, HyperParams);
acc = ComputeAccuracy_BN(P, y_test);
%}

function s_hat = BatchNormalize(s, mu, v)
s_hat = diag((v + eps).^(-1/2)) * (s - mu);
end

function G_batch = BatchNormBackPass(G_batch, S_batch, mu, v)
%Steps are listed on assignment
n = size(S_batch, 2);

sigma1 = ((v + eps).^(-1/2))';
sigma2 = ((v + eps).^(-3/2))';

G_1 = G_batch.*(sigma1'*ones(1, n));
G_2 = G_batch.*(sigma2'*ones(1, n));

D = S_batch - mu * ones(1, n);
c = (G_2.*D) * ones(n, 1);

G_batch = G_1 - 1/n * (G_1 * ones(n, 1)) * ones(1, n) - 1/n * D .*(c * ones(1, n));
end

function acc = ComputeAccuracy(X, y, HyperParams)

P = EvaluateClassifier(X, HyperParams);
[~, index] = max(P);
acc = sum(index == y)/length(P);
end

function acc = ComputeAccuracy_BN(P, y)

[~, index] = max(P);
acc = sum(index == y)/length(P);
end

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

end

function [c, l] = ComputeCost_BN(X, Y, HyperParams, lambda, varargin)

W = HyperParams.W;
b = HyperParams.b;
len = size(Y,2);
H{1} = X;
sumW = sum(sum(W{1}.^2));
k = numel(W);

for i = 1:k - 1
    if HyperParams.use_bn
        gamma = HyperParams.gamma{i};
        beta = HyperParams.beta{i};
        
        s = W{i} * H{i} + b{i};
        
        if nargin == 6
            mu = varargin{1}(i);
            v = varargin{2}(i);
        else
            [~, n] = size(s);
            mu = mean(s, 2);
            v = var(s, 0, 2);
            v = v * (n - 1)/n;
        end
        
        s_hat = BatchNormalize(s, mu, v);
        s_tilde = gamma .* s_hat + beta;
        H{i + 1} = max(0, s_tilde);
        
    else
        s = W{i} * H{i} + b{i};
        H{i + 1} = max(s, 0);
    end
    sumW = sumW + sum(sum(W{i+1}.^2));
end

P = softmax(W{k} * H{k} + b{k});

l = -sum(log(sum(Y .* P, 1)))/len;
c = l + lambda * sumW;   

end

function Grads = ComputeGradients(X, Y, HyperParams, lambda)

W = HyperParams.W;
b = HyperParams.b;
k = numel(W);
H{1} = X;
len = size(Y,2);
a = 0.01;

%forward propagation
for i = 1:k-1
    if HyperParams.use_bn %Batch normalization version
        %This is to implement the forward pass pf BN
        gamma = HyperParams.gamma{i};
        beta = HyperParams.beta{i};
        s{i} = W{i} * H{i} + b{i};
        
        
        %calculate mean and variance of score s
        mu{i} = mean(s{i}, 2);
        v{i} = var(s{i}, 0, 2) * (size(s{i}, 2) - 1)/size(s{i}, 2);
        %Apply batch normalizetion in hints
        s_hat{i} = BatchNormalize(s{i}, mu{i}, v{i});
        s_tilde = gamma .* s_hat{i} + beta;
        H{i + 1} = max(0, s_tilde); %relu
        %H{i + 1} = max(a * s_tilde, s_tilde); %leaky relu
        
        %{
        %new code for BN after Relu
        relu_s{i} = max(0, s{i});
        mu{i} = mean(relu_s{i}, 2);
        v{i} = var(relu_s{i}, 0, 2) * (size(relu_s{i}, 2) - 1)/size(relu_s{i}, 2);
        s_hat{i} = BatchNormalize(relu_s{i}, mu{i}, v{i});
        H{i + 1} = gamma .* s_hat{i} + beta;
        %}
        
    else
        s{i} = W{i} * H{i} + b{i};
        H{i+1} = max(s{i}, 0);
    end
end
P = softmax(W{k}*H{k} + b{k});

%backward propagation
G = -(Y-P);

if HyperParams.use_bn % Batch normalization version
    Grads.mu = mu;
    Grads.v = v;
    
    grad_W{k} = (1/len) * G * H{k}' + 2 * lambda * W{k};
    grad_b{k} = (1/len) * G * ones(len, 1);
    
    G = W{k}'*G;
    G = G.*sign(H{k}); %for BN after relu, this should be comment
    
    %backpropagate for leaky relu
    %tmp_sign_k = sign(H{k});
    %tmp_sign_k(tmp_sign_k < 0) = a;
    %G = G .* tmp_sign_k;
   
    
    tmp = k - 1;
    for i = 1:k-1
        %steps shown on assignment pdf
        grad_gamma{tmp} = 1/len * (G .* s_hat{tmp}) * ones(len, 1);
        grad_beta{tmp} = 1/len * G * ones(len, 1);
        
        
        G = G .*(HyperParams.gamma{tmp} * ones(1, len));
        G = BatchNormBackPass(G, s{tmp}, mu{tmp}, v{tmp});
        
        grad_W{tmp} = (1/len) * G * H{tmp}' + 2 * lambda * W{tmp};
        grad_b{tmp} = (1/len) * G * ones(len, 1);
        if tmp > 1
            G = W{tmp}' * G;
            G = G .* sign(H{tmp});
            
            %backpropagate for leaky relu
            %tmp_sign = sign(H{tmp});
            %tmp_sign(tmp_sign < 0) = a;
            %G = G .* tmp_sign;
            
        end
        tmp = tmp - 1;
        
        %{
        %new code for BN after relu
        G = G .*(HyperParams.gamma{tmp} * ones(1, len));
        G = BatchNormBackPass(G, relu_s{tmp}, mu{tmp}, v{tmp});
        
        %bonus part added
        G = G .* sign(relu_s{tmp}); % added for bonus part
        
        
        grad_W{tmp} = (1/len) * G * H{tmp}' + 2 * lambda * W{tmp};
        grad_b{tmp} = (1/len) * G * ones(len, 1);
        if tmp > 1
            G = W{tmp}' * G;
            %code removed for bonus part
            %G = G .* sign(H{tmp});
        end
        tmp = tmp - 1;
        %}
    end
    
Grads.gamma = grad_gamma;
Grads.beta = grad_beta;

else
    count = k;
    for i = 2:k
        grad_W{count} = (1/len) * G * H{count}' + 2 * lambda * W{count};
        grad_b{count} = (1/len) * G * ones(len, 1);
        G = W{count}' * G;
        G = G .* sign(H{count});
        count = count - 1;
    end
    grad_W{1} = (1/len) * G * X' + 2 * lambda .* W{1};
    grad_b{1} = (1/len) * G * ones(len, 1);
end

Grads.W = grad_W;
Grads.b = grad_b;


end

function P = EvaluateClassifier(X, HyperParams)

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
        mu = mean(s, 2);
        v = var(s, 0, 2) * (size(s, 2) - 1)/size(s, 2);
        
        %Apply batch normalizetion in hints
        s_hat = BatchNormalize(s, mu, v);
        s_t = gamma .* s_hat + beta;
        H{i + 1} = max(0, s_t);
        
    else
        s = W{i} * H{i} + b{i};
        H{i + 1} = max(0, s);
    end
end
P = softmax(W{k} * H{k} + b{k});

end

function P = EvaluateClassifier_BN(X, HyperParams, varargin)

W = HyperParams.W;
b = HyperParams.b;
k = numel(W);
H{1} = X;
a = 0.01;

for i = 1:k-1
    if HyperParams.use_bn
        %This is to implement the forward pass pf BN
        gamma = HyperParams.gamma{i};
        beta = HyperParams.beta{i};
        s = W{i} * H{i} + b{i};
        
        
        %original code
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
        H{i + 1} = max(0, s_tilde); %relu
        %H{i + 1} = max(a * s_tilde, s_tilde); %leaky_relu
        
        %{
        %new code for BN after Relu
        relu_s = max(0, s);
        mu = mean(relu_s, 2);
        v = var(relu_s, 0, 2) * (size(relu_s, 2) - 1)/size(s, 2);
        s_hat = BatchNormalize(relu_s, mu, v);
        H{i + 1} = gamma .* s_hat + beta;
        %}
        
    else
        s = W{i} * H{i} + b{i};
        H{i + 1} = max(0, s);
    end
end
P = softmax(W{k} * H{k} + b{k});

end


function Adampara = InitialAdam(HyperParams)
k = length(HyperParams.W);
S_W = cell(1, k); 
S_b = cell(1, k); 
S_gamma = cell(1, k); 
S_beta = cell(1, k); 
for i = 1:k
    S_W{i} = HyperParams.W{i}*0;
    S_b{i} = HyperParams.b{i}*0; 
    S_gamma{i} = HyperParams.gamma{i}*0; 
    S_beta{i} = HyperParams.beta{i}*0; 
end
Adampara.mW = S_W;
Adampara.mb = S_b;
Adampara.mbetas = S_beta;
Adampara.mgammas = S_gamma;
Adampara.vW = S_W;
Adampara.vb = S_b;
Adampara.vbetas = S_beta;
Adampara.vgammas = S_gamma;
end

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

end

function [X, Y, y] = LoadBatch(filename)

A = load(filename);
X = double(A.data')/255; % normalize

tmp = double(A.labels');
y = tmp + ones(1, size(A.labels, 1)); %adapt the index to MATLAB

Y = zeros(10, size(A.labels, 1));
for i = 1:size(A.labels, 1)
    Y(y(i),i) = 1;
end

end

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