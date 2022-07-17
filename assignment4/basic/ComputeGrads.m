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