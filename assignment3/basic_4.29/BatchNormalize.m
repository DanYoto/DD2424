function s_hat = BatchNormalize(s, mu, v)
s_hat = diag((v + eps).^(-1/2)) * (s - mu);
end