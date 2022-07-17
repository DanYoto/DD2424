addpath '/Users/yotod/DD2424/MATLAB version/Datasets/cifar-10-batches-mat';

%{
%The first method - more training data


%parameter setting
lambda = 0;
n_batch = 100;
eta = 0.001;
n_epochs = 40;

%Load the data
[X_train_1, Y_train_1, y_train_1] = LoadBatch('data_batch_1.mat');
[X_train_2, Y_train_2, y_train_2] = LoadBatch('data_batch_2.mat');
[X_train_3, Y_train_3, y_train_3] = LoadBatch('data_batch_3.mat');
[X_train_4, Y_train_4, y_train_4] = LoadBatch('data_batch_4.mat');
[X_train_5, Y_train_5, y_train_5] = LoadBatch('data_batch_5.mat');

X_train = [X_train_1, X_train_2, X_train_3, X_train_4, X_train_5];
Y_train = [Y_train_1, Y_train_2, Y_train_3, Y_train_4, Y_train_5];
y_train = [y_train_1, y_train_2, y_train_3, y_train_4, y_train_5];


X_val = X_train(:, 1:1000);
Y_val = Y_train(:, 1:1000);
y_val = y_train(:, 1:1000);

X_train = X_train(:, 1001:end);
Y_train = Y_train(:, 1001:end);


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


%training
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
%}


%{
%The second method is about shuffle the training data
%parameter setting
lambda = 0;
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

[Wstar, bstar] = MiniBatchGD_rand(X_train, Y_train, X_val, Y_val, n_batch, eta, n_epochs, W, b, lambda);

%calculate the accuracy of test data
accuracy = ComputeAccuracy(X_test, y_test, Wstar, bstar);

%visualiza the weight matrix 
for i = 1:10
    im = reshape(Wstar(i,:), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
montage(s_im, 'Size', [1,10])
%}



%{
%The third method is to decay the learning rate
%parameter setting
lambda = 0;
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


[Wstar, bstar] = MiniBatchGD_decay(X_train, Y_train, X_val, Y_val, n_batch, eta, n_epochs, W, b, lambda);

%calculate the accuracy of test data
accuracy = ComputeAccuracy(X_test, y_test, Wstar, bstar);

%visualiza the weight matrix 
for i = 1:10
    im = reshape(Wstar(i,:), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
montage(s_im, 'Size', [1,10])
%}

%{
% The fourth method is to increase epochs
%parameter setting
lambda = 0;
n_batch = 100;
eta = 0.001;
n_epochs = 50;

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
%}



%part 2
%parameter setting
lambda = 0.1;
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

%P = EvaluateClassifier_bonus(X_train, W, b);

[Wstar, bstar] = MiniBatchGD_bonus(X_train, Y_train, X_val, Y_val, y_train, y_val, n_batch, eta, n_epochs, W, b, lambda);

%calculate the accuracy of test data
accuracy = ComputeAccuracy(X_test, y_test, Wstar, bstar);

P = EvaluateClassifier_bonus(X_test, Wstar, bstar);
[~, index] = max(P);
tmp = zeros(1, 20);
for i = 1:10000
    if index(i) == y_test(i)
        tmp(y_test(i)) = tmp(y_test(i)) + 1;
    else
        tmp(y_test(i)+10) = tmp(y_test(i)+10) + 1;
    end
end
tmp = tmp /10000;

%visualiza the weight matrix 
%for i = 1:10
%    im = reshape(Wstar(i,:), 32, 32, 3);
%    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
%    s_im{i} = permute(s_im{i}, [2, 1, 3]);
%end
%montage(s_im, 'Size', [1,10])
