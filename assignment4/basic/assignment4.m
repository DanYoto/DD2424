
%read in the contents of the text file
book_name = 'goblet_book.txt';
fid = fopen(book_name, 'r');
book_data = fscanf(fid, '%c');
fclose(fid);

book_chars = unique(book_data);
K = size(book_chars, 2);

char_to_ind = containers.Map('KeyType', 'char', 'ValueType', 'int32');
ind_to_char = containers.Map('KeyType', 'int32', 'ValueType', 'char');

%creating dictionary to map the characters and index
for i = 1:K
    char_to_ind(book_chars(i)) = i;
    ind_to_char(i) = book_chars(i);
end


%set hyper_parameters and initialize RNN
m = 100; %hidden state defined in the pdf version
eta = 0.1;
seq_length = 25;%length of character
sig = 0.01;

RNN.b = zeros(m, 1);
RNN.c = zeros(K, 1);
RNN.U = randn(m, K) * sig;
RNN.W = randn(m, m) * sig;
RNN.V = randn(K, m) * sig;

%{
%check gradients computation
X_chars = book_data(1:seq_length);
Y_chars = book_data(2:seq_length+1);
X = zeros(K, seq_length);
Y = zeros(K, seq_length);
h0 = zeros(m, 1);

%one hot encoding
for i = 1:seq_length
    x_idx = char_to_ind(X_chars(i)); 
    X(x_idx, i) = 1;
    y_idx = char_to_ind(Y_chars(i)); 
    Y(y_idx, i) = 1;
end

Grads_my = ComputeGrads(X, Y, RNN, h0);
Grads_num = ComputeGradsNum(X, Y, RNN, 1e-5);
%absolute difference error
absolute_error_b = max(max(abs(Grads_my.b - Grads_num.b)));
absolute_error_c = max(max(abs(Grads_my.c - Grads_num.c)));
absolute_error_U = max(max(abs(Grads_my.U - Grads_num.U)));
absolute_error_W = max(max(abs(Grads_my.W - Grads_num.W)));
absolute_error_V = max(max(abs(Grads_my.V - Grads_num.V)));
%}


%train RNN using AdaGrad
smooth_loss = 0;
e = 1;
n_epochs = 7;
iter = n_epochs * floor(length(book_data)/seq_length);
loss_min = inf;

%Tmp is used to store the sum of each gradients
for f = fieldnames(RNN)'
    tmp.(f{1}) = zeros(size(RNN.(f{1})));
end


for t = 1:iter
    %This is the input characters
    X_chars = book_data(e : e + seq_length - 1);
    %This is the labelled characters
    Y_chars = book_data(e + 1:e + seq_length);
    
    %next we need to convert the input characters to one-hot encoding
    %matrix X and Y
    X = zeros(K, seq_length);
    Y = zeros(K, seq_length);
    for i = 1:seq_length
        x_idx = char_to_ind(X_chars(i));
        X(x_idx, i) = 1;
        y_idx = char_to_ind(Y_chars(i));
        Y(y_idx, i) = 1;
    end
    
    %define hprev
    if e == 1
        hprev = zeros(m, 1);
    else
        hprev = h(:, seq_length);
    end
    
    %compute the gradients
    [grads, h, loss] = ComputeGrads(X, Y, RNN, hprev);
    
    %compute smooth loss by applying AdaGrad
    if t == 1
        smooth_loss = loss;
    end
    smooth_loss = 0.999 * smooth_loss + 0.001 * loss;
    stored_loss(t) = smooth_loss;
    
    %to store the best parameter
    if loss_min > smooth_loss
        RNNstar = RNN;
        h0star = hprev;
        x0star = X(:, 1);
        loss_min = smooth_loss;
    end
    
    if t == 1 || mod(t, 10000) == 0
        disp(t);
        disp(smooth_loss);
        text = SynthesizeText(RNN, hprev, X(:, 1), 200, ind_to_char);
        disp(char(text));
    end
    
    %Implementation of AdaGrad
    for f = fieldnames(RNN)'
        tmp.(f{1}) = tmp.(f{1}) + grads.(f{1}).^2;
        RNN.(f{1}) = RNN.(f{1}) - eta * (grads.(f{1}) ./(tmp.(f{1}) + eps).^(0.5));
    end
    

    e = e + seq_length;
    %to find if e is at the end of the book or not
    if e > length(book_data) - seq_length - 1
        e = 1;
    end
    

end

t = 1:iter;
plot(t, stored_loss);
xlabel('update srep');
ylabel('loss');

disp("---------------------")
disp('Best model')
text = SynthesizeText(RNNstar, h0star, x0star, 1000, ind_to_char);
disp(char(text));

function [grads, h, loss] = ComputeGrads(X, Y, RNN, h0)

    n = size(X,2); 
    m = size(RNN.b,1);
    
    h = zeros(m,n+1);
    h(:,1) = h0; 
    
    % forward pass
    for t = 1:n
        a{t} = RNN.W * h(:,t) + RNN.U * X(:,t) + RNN.b;
        h(:,t+1) = tanh(a{t});
        o{t} = RNN.V * h(:,t+1) + RNN.c;
        P(:,t) = softmax(o{t});  
    end
    
    % backpropagation
    G = -(Y-P); 
    grads_o = G';
    
    loss = -sum(log(sum(Y .* P, 1)));
    
    grads_h = zeros(n,m);
    grads_a = zeros(n,m);
    grads_h(n,:) = grads_o(n,:) * RNN.V;
    grads_a(n,:) = grads_h(n,:) * diag(1-(tanh(a{n}).^2));
    
    for t = n-1:-1:1
        grads_h(t,:) = grads_o(t,:) * RNN.V + grads_a(t + 1,:) * RNN.W;
        grads_a(t,:) = grads_h(t,:) * diag(1-tanh(a{t}).^2);
    end
    
    %find the derivative 
    grads.b = grads_a' * ones(n,1);
    grads.c = G * ones(n,1);
    grads.U = grads_a' * X';
    grads.W = grads_a' * h(:,1:n)';
    grads.V = grads_o' * h(:,2:n+1)';

    %clip the gradients
    for f = fieldnames(grads)'
        grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
    end

end

function loss = ComputeLoss(X, Y, RNN, h0)

    n = size(X,2); 
    m = size(RNN.b,1);
    h = zeros(m,n+1);
    h(:,1) = h0; 

    for t = 1:n
        at = RNN.W * h(:,t) + RNN.U * X(:,t) + RNN.b;
        h(:,t+1) = tanh(at); 
        ot = RNN.V * h(:,t + 1) + RNN.c;
        P(:,t) = softmax(ot);   
    end
    
    loss = -sum(log(sum(Y .* P, 1)));
end

function text = SynthesizeText(RNN, h0, x0, n, ind_to_char)

ht = h0;
xt = x0;
K = size(RNN.U, 2);

for i = 1:n
    at = RNN.W * ht + RNN.U * xt + RNN.b;
    ht = tanh(at);
    ot = RNN.V * ht + RNN.c;
    pt = softmax(ot);
    
    cp = cumsum(pt);
    a = rand;
    ixs = find(cp - a > 0);
    ii = ixs(1);
    
    xt = zeros(K, 1);
    xt(ii, 1) = 1;
    text(i) = ind_to_char(ii);
end


end

function num_grads = ComputeGradsNum(X, Y, RNN, h)

for f = fieldnames(RNN)'
    disp('Computing numerical gradient for')
    disp(['Field name: ' f{1} ]);
    num_grads.(f{1}) = ComputeGradNumSlow(X, Y, f{1}, RNN, h);
end
end

function grad = ComputeGradNumSlow(X, Y, f, RNN, h)

n = numel(RNN.(f));
grad = zeros(size(RNN.(f)));
hprev = zeros(size(RNN.W, 1), 1);
for i=1:n
    RNN_try = RNN;
    RNN_try.(f)(i) = RNN.(f)(i) - h;
    l1 = ComputeLoss(X, Y, RNN_try, hprev);
    RNN_try.(f)(i) = RNN.(f)(i) + h;
    l2 = ComputeLoss(X, Y, RNN_try, hprev);
    grad(i) = (l2-l1)/(2*h);
end
end