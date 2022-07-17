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