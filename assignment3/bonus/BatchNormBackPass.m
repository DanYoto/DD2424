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