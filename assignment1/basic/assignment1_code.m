addpath '/Users/yotod/DD2424/Datasets/cifar-10-batches-mat';

%parameter setting
lambda = 1;
n_batch = 100;
eta = 0.001;
n_epochs = 40;

%Load the data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

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
W = 0.01 * randn(K, d);
b = 0.01 * randn(K, 1);

%{
%probability that evaluate the network function of subsets
P = EvaluateClassifier(X_train, W, b);
%At this part we should compare the analytical and numerical gradient
lambda = 0.1;
%my result
samples = 100;
[mygrad_W, mygrad_b] = ComputeGradients(X_train(:,samples), Y_train(:,samples), P(:,samples), W, lambda);
%theoretical
[grad_b, grad_W] = ComputeGradsNumSlow(X_train(:,samples), Y_train(:,samples), W, b, lambda, 1e-6);

%relative error method
error_W = abs(norm(grad_W)-norm(mygrad_W)) ./ max(eps, norm(abs(grad_W))+norm(abs(mygrad_W)));
error_b = abs(norm(grad_b)-norm(mygrad_b)) ./ max(eps, norm(abs(grad_b))+norm(abs(mygrad_b)));
%}


[Wstar, bstar] = MiniBatchGD(X_train, Y_train, X_val, Y_val, n_batch, eta, n_epochs, W, b, lambda);

%calculate the accuracy of test data
accuracy = ComputeAccuracy(X_test, y_test, Wstar, bstar);

%visualiza the weight matrix 
for i = 1:10
    im = reshape(Wstar(i,:), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
montage(s_im, 'Size', [1,10])

%function compute accuracy
function acc = ComputeAccuracy(X, y, W, b)

P = EvaluateClassifier(X, W, b);
[~, index] = max(P);
acc = sum(index == y)/length(P);

end

%function compute cost
function J = ComputeCost(X, Y, W, b, lambda)

P = EvaluateClassifier(X, W, b);

%calculate the number of samples
D = size(P, 2);

J = -sum(log(sum(Y .* P, 1))) / D + lambda * sum(sum(W.^2));

end

%function compute gradients
%function compute accuracy
function acc = ComputeAccuracy(X, y, W, b)

P = EvaluateClassifier(X, W, b);
[~, index] = max(P);
acc = sum(index == y)/length(P);

end

%function compute cost
function J = ComputeCost(X, Y, W, b, lambda)

P = EvaluateClassifier(X, W, b);

%calculate the number of samples
D = size(P, 2);

J = -sum(log(sum(Y .* P, 1))) / D + lambda * sum(sum(W.^2));

end

%function compute gradientsnum
function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

c = ComputeCost(X, Y, W, b, lambda);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c) / h;
end

for i=1:numel(W)   
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c) / h;
end
end

%function compute gradientsnumslow
function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) - h;
    c1 = ComputeCost(X, Y, W, b_try, lambda);
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c1) / (2*h);
end

for i=1:numel(W)
    
    W_try = W;
    W_try(i) = W_try(i) - h;
    c1 = ComputeCost(X, Y, W_try, b, lambda);
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c1) / (2*h);
end
end

%function compute loss
function J = ComputeLoss(X, Y, W, b)

P = EvaluateClassifier(X, W, b);

%calculate the number of samples
D = size(P, 2);

J = -sum(log(sum(Y .* P, 1))) / D;

end

%function evaluateclassifier
function P = EvaluateClassifier(X, W, b)

s = W * X + b;
P = softmax(s);

end

%function loadbatch
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

%function minibatch
function [Wstar, bstar] = MiniBatchGD(X_train, Y_train, X_val, Y_val, n_batch, eta, n_epochs, W, b, lambda)
train_cost = zeros(n_epochs, 1);
val_cost = zeros(n_epochs, 1);
train_loss = zeros(n_epochs, 1);
val_loss = zeros(n_epochs, 1);
for epoch = 1:n_epochs
    for batch = 1:size(X_train, 2) / n_batch
        j_start = (batch - 1) * n_batch + 1;
        j_end = batch * n_batch;
        X_tr_ba = X_train(:, j_start:j_end);
        Y_tr_ba = Y_train(:, j_start:j_end);
        
        P = EvaluateClassifier(X_tr_ba, W, b);
        [grad_W, grad_b] = ComputeGradients(X_tr_ba, Y_tr_ba, P, W, lambda);
        W = W - eta * grad_W;
        b = b - eta * grad_b;
    end
    train_cost(epoch) = ComputeCost(X_train, Y_train, W, b, lambda);
    val_cost(epoch) = ComputeCost(X_val, Y_val, W, b, lambda);
    train_loss(epoch) = ComputeLoss(X_train, Y_train, W, b);
    val_loss(epoch) = ComputeLoss(X_val, Y_val, W, b);
end
Wstar = W;
bstar = b;

figure
plot(1:n_epochs, train_cost);
hold on
plot(1:n_epochs, val_cost);
xlabel('epoch');
ylabel('cost');
legend('training cost', 'validation cost');
hold off

figure
plot(1:n_epochs, train_loss);
hold on
plot(1:n_epochs, val_loss);
xlabel('epoch');
ylabel('loss');
legend('training loss', 'validation loss');
hold off
end