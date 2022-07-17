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

%{
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
%}
end
