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