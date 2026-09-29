function C = ref_tprod(A, B)
% Independent reference: FFT along dims 3..p, slice-wise product, plain IFFT.
sa = size(A); sb = size(B); p = numel(sa);
Ah = A; Bh = B;
for d = 3:p, Ah = fft(Ah, [], d); Bh = fft(Bh, [], d); end
n = prod(sa(3:end));
Ch = zeros([sa(1) sb(2) n]);
for k = 1:n, Ch(:,:,k) = Ah(:,:,k) * Bh(:,:,k); end
C = reshape(Ch, [sa(1) sb(2) sa(3:end)]);
for d = 3:p, C = ifft(C, [], d); end
if isreal(A) && isreal(B), C = real(C); end
end
