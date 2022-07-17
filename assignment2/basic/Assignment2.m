addpath '/Users/yotod/DD2424/MATLAB version/Datasets/cifar-10-batches-mat';

%Load the data
[X_train_1, Y_train_1, y_train_1] = LoadBatch('data_batch_1.mat');
[X_train_2, Y_train_2, y_train_2] = LoadBatch('data_batch_2.mat');
[X_train_3, Y_train_3, y_train_3] = LoadBatch('data_batch_3.mat');
[X_train_4, Y_train_4, y_train_4] = LoadBatch('data_batch_4.mat');
[X_train_5, Y_train_5, y_train_5] = LoadBatch('data_batch_5.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

X_train = [X_train_1, X_train_2, X_train_3, X_train_4, X_train_5];
Y_train = [Y_train_1, Y_train_2, Y_train_3, Y_train_4, Y_train_5];
y_train = [y_train_1, y_train_2, y_train_3, y_train_4, y_train_5];

X_val = X_train(:, 1:1000);
Y_val = Y_train(:, 1:1000);
y_val = y_train(:, 1:1000);

X_train = X_train(:, 1001:end);
Y_train = Y_train(:, 1001:end);
y_train = y_train(:, 1001:end);


%parameter setting
lambda = 3.04e-4;
n_batch = 100;
n = size(X_train, 2);
n_s = 2 * floor(n/n_batch);
eta_min = 1e-5;
eta_max = 1e-1;
n_cycle = 4;
n_epochs = 2 * n_s * n_cycle * n_batch / n;
t = 1;
l = 0;
count = 0;

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
K = size(Y_train, 1);
d = size(X_train, 1);
m = 50;
[W, b] = InitialParams(K, d, m);

%check the gradients work or not
%[grad_mW, grad_mb] = ComputeGradients(X_train(:,1:1000), Y_train(:,1:1000), W, b, lambda);
%[grad_b, grad_W] = ComputeGradsNum(X_train(:,1:1000), Y_train(:,1:1000), W, b, lambda, 1e-6);
%relative_error = max(max(abs(grad_W{1} - grad_mW{1})));


train_cost = zeros(2 * n_s, 1);
val_cost = zeros(2 * n_s, 1);
train_loss = zeros(2 * n_s, 1);
val_loss = zeros(2 * n_s, 1);
train_acc = zeros(2 * n_s, 1);
val_acc = zeros(2 * n_s, 1);

for epoch = 1:n_epochs
    for batch = 1:size(X_train, 2) / n_batch

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

        [W, b] = MiniBatchGD(X_tr_ba, Y_tr_ba, W, b, lambda, eta);
        
        if mod(t, 10) == 1
            [train_cost(t), train_loss(t)] = ComputeCost(X_train, Y_train, W, b, lambda);
            [val_cost(t), val_loss(t)] = ComputeCost(X_val, Y_val, W, b, lambda);
            train_acc(t) = ComputeAccuracy(X_train, y_train, W, b);
            val_acc(t) = ComputeAccuracy(X_val, y_val, W, b);
        end
        

        t = t + 1;

        
        if t > 2 * (l + 1) * n_s
            l = l + 1;
        end
    end

end

Wstar = W;
bstar = b;
acc = ComputeAccuracy(X_test, y_test, Wstar, bstar);


figure
plot(1:10:2*n_cycle*n_s, train_cost(1:10:end));
hold on
plot(1:10:2*n_cycle*n_s, val_cost(1:10:end));
xlabel('update step');
ylabel('cost');
legend('training cost', 'validation cost');
hold off

figure
plot(1:10:2*n_cycle*n_s, train_loss(1:10:end));
hold on
plot(1:10:2*n_cycle*n_s, val_loss(1:10:end));
xlabel('update step');
ylabel('loss');
legend('training loss', 'validation loss');
hold off

figure
plot(1:10:2*n_cycle*n_s, train_acc(1:10:end));
hold on
plot(1:10:2*n_cycle*n_s, val_acc(1:10:end));
xlabel('update step');
ylabel('accuracy');
legend('training accuracy', 'validation accuracy');
hold off


%calculate the accuracy of test data
accuracy = ComputeAccuracy(X_test, y_test, Wstar, bstar);


%visualiza the weight matrix 
%for i = 1:10
%    im = reshape(Wstar(i,:), 32, 32, 3);
%    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
%    s_im{i} = permute(s_im{i}, [2, 1, 3]);
%end
%montage(s_im, 'Size', [1,10])


%{
%coarse search 
%Load the data
[X_train_1, Y_train_1, y_train_1] = LoadBatch('data_batch_1.mat');
[X_train_2, Y_train_2, y_train_2] = LoadBatch('data_batch_2.mat');
[X_train_3, Y_train_3, y_train_3] = LoadBatch('data_batch_3.mat');
[X_train_4, Y_train_4, y_train_4] = LoadBatch('data_batch_4.mat');
[X_train_5, Y_train_5, y_train_5] = LoadBatch('data_batch_5.mat');

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
%lambda = 0.01;
n_batch = 100;
n = size(X_train, 2);
n_s = 2 * floor(n/n_batch);
eta_min = 1e-5;
eta_max = 1e-1;
n_cycle = 1;
n_epochs = 2 * n_s * n_cycle * n_batch / n;
t = 1;
l = 0;

%pre-process raw data
mean_X = mean(X_train, 2);
std_X = std(X_train, 0, 2);
X_train = X_train - repmat(mean_X, [1, size(X_train, 2)]);
X_train = X_train ./ repmat(std_X, [1, size(X_train, 2)]);

X_val = X_val - repmat(mean_X, [1, size(X_val, 2)]);
X_val = X_val ./ repmat(std_X, [1, size(X_val, 2)]);

%give random value to lambda
l_min = -5;
l_max = -1;
l_tmp = l_min + (l_max - l_min) * rand(1, 8);
lamb = 10.^l_tmp;
len = size(lamb, 2);
acc = zeros(1, 8);


for i = 1:8
%initializing paraneter W and d
K = size(Y_train, 1);
d = size(X_train, 1);
m = 50;
[W, b] = InitialParams(K, d, m);

    for epoch = 1:n_epochs
        for batch = 1:size(X_train, 2) / n_batch
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

            [W, b] = MiniBatchGD(X_tr_ba, Y_tr_ba, W, b, lamb(i), eta);
        
            t = t + 1;
        
            if t > 2 * (l + 1) * n_s
                l = l + 1;
            end
        end

    end

Wstar = W;
bstar = b;
acc(i) = ComputeAccuracy(X_val, y_val, Wstar, bstar);

end
%}




%{
%by coarse search, we could find the range between 2.14e-4 - 8.25e-4

%random search
%-2.7 - -6.6
%Load the data
[X_train_1, Y_train_1, y_train_1] = LoadBatch('data_batch_1.mat');
[X_train_2, Y_train_2, y_train_2] = LoadBatch('data_batch_2.mat');
[X_train_3, Y_train_3, y_train_3] = LoadBatch('data_batch_3.mat');
[X_train_4, Y_train_4, y_train_4] = LoadBatch('data_batch_4.mat');
[X_train_5, Y_train_5, y_train_5] = LoadBatch('data_batch_5.mat');

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
%lambda = 0.01;
n_batch = 100;
n = size(X_train, 2);
n_s = 2 * floor(n/n_batch);
eta_min = 1e-5;
eta_max = 1e-1;
n_cycle = 1;
n_epochs = 2 * n_s * n_cycle * n_batch / n;
t = 1;
l = 0;

%pre-process raw data
mean_X = mean(X_train, 2);
std_X = std(X_train, 0, 2);
X_train = X_train - repmat(mean_X, [1, size(X_train, 2)]);
X_train = X_train ./ repmat(std_X, [1, size(X_train, 2)]);

X_val = X_val - repmat(mean_X, [1, size(X_val, 2)]);
X_val = X_val ./ repmat(std_X, [1, size(X_val, 2)]);

%give random value to lambda
%l_min = -5;
%l_max = -1;
%lamb = zeros(1, 8);
l_min = -5;
l_max = -1;
l_tmp = l_min + (l_max - l_min) * rand(1, 8);
lamb = 10.^l_tmp;
%len = size(lamb, 2);
acc = zeros(1, 62);

lamb = 2.14e-4:0.00001:8.25e-4;

for i = 1:62
%initializing paraneter W and d
K = size(Y_train, 1);
d = size(X_train, 1);
m = 50;
[W, b] = InitialParams(K, d, m);

    for epoch = 1:n_epochs
        for batch = 1:size(X_train, 2) / n_batch
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

            [W, b] = MiniBatchGD(X_tr_ba, Y_tr_ba, W, b, lamb(i), eta);
        
            t = t + 1;
        
            if t > 2 * (l + 1) * n_s
                l = l + 1;
            end
        end

    end

Wstar = W;
bstar = b;
acc(i) = ComputeAccuracy(X_val, y_val, Wstar, bstar);

end
%optimal lambda 3.04e-4
%}

function acc = ComputeAccuracy(X, y, W, b)

P = EvaluateClassifier(X, W, b);
[~, index] = max(P);
acc = sum(index == y)/length(P);

end

function [c, l] = ComputeCost(X, Y, W, b, lambda)

    D = size(Y,2); 
    W_1 = W{1};
    W_2 = W{2}; 
    b_1 = b{1}; 
    b_2 = b{2};
    
    H = max(W_1*X+b_1,0); 
    P = softmax(W_2*H+b_2); 
    
    l = -sum(log(sum(Y .* P, 1)))/D;
    c = l + lambda*(sum(sum(W_1.^2))+sum(sum(W_2.^2)));
    

end

function [grad_W, grad_b] = ComputeGradients(X, Y, W, b, lambda)

W_1 = W{1};
W_2 = W{2};
b_1 = b{1};
b_2 = b{2};
s_1 = W_1 * X + b_1;
H = max(s_1, 0);

P_2 = EvaluateClassifier(X, W, b);

%back propagation
%2nd layer
len_2 = size(H, 2);
G = -(Y - P_2)';
grad_W_2 = (1/len_2) * G' * H' + 2 * lambda .* W_2;
grad_b_2 = (1/len_2) * G' * ones(len_2, 1);

%1st layer
G = G * W_2;
G = G' .* sign(H);
len_1 = size(X, 2);
grad_W_1 = (1/len_1) * G * X' + 2 * lambda .* W_1;
grad_b_1 = (1/len_1) * G * ones(len_1, 1);

grad_W = {grad_W_1, grad_W_2};
grad_b = {grad_b_1, grad_b_2};

end

function P_2 = EvaluateClassifier(X, W, b)

W_1 = W{1};
W_2 = W{2};
b_1 = b{1};
b_2 = b{2};
s_1 = W_1 * X + b_1;
H = max(s_1, 0);

s_2 = W_2 * H + b_2;
P_2 = softmax(s_2);

end

function [W, b] = InitialParams(K, d, m)
W_1 = 1/sqrt(d) * randn(m, d);
b_1 = zeros(m,1);
W_2 = 1/sqrt(m) * randn(K, m);
b_2 = zeros(K,1);
W = {W_1, W_2};
b = {b_1, b_2};
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

function [Wstar, bstar] = MiniBatchGD(X, Y, W, b, lambda, eta)

W_1 = W{1};
W_2 = W{2};
b_1 = b{1};
b_2 = b{2};

[grad_W, grad_b] = ComputeGradients(X, Y, W, b, lambda);

W_1 = W_1 - eta * grad_W{1};
W_2 = W_2 - eta * grad_W{2};
b_1 = b_1 - eta * grad_b{1};
b_2 = b_2 - eta * grad_b{2};

Wstar = {W_1, W_2};
bstar = {b_1, b_2};
end