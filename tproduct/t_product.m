function C = t_product(A,B)
% T_PRODUCT  t-product of two p-order tensors by definition,
%   C = fold(bcirc(A) * unfold(B)), applied recursively for p > 3.
%   Slow reference implementation; use T_PRODUCT_FFT in practice.
if numel(size(A))==3
    C = fold(bcirc(A)*unfold(B),size(A,3));
else
C = fold(t_product(bcirc(A),unfold(B)),size(A,numel(size(A))));
end