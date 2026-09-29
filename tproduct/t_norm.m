function tv2 = t_norm(X)
% T_NORM  Tensor magnitude |X| = (X^T * X)^(1/2) under the t-product,
%   computed slice-wise in the Fourier domain.
tsize = size(X);
M = length(tsize);
num_slices = prod(tsize(3:end));
X_hat = X;
tv2  = zeros([tsize(2) tsize(2) tsize(3:end)]);
for i = 3:M
    X_hat = fft(X_hat,[],i);
end
for i = 1:num_slices
    tv2(:,:,i) = sqrtm(X_hat(:,:,i)'*X_hat(:,:,i));
end
for i = M:-1:3
    tv2 = ifft(tv2,[],i);
end
end