function A_inverse = t_inverse(A)
% this is a function to get the inverse of a p-order tensor
% the input is a p-order tensor A
% the output is the inverse tensor A_inverse

tsize = size(A);
p = length(tsize);
num_slices = 1;
for i = 3:p
    num_slices = num_slices*tsize(i);
end

for i = 3 : p
    A = fft(A,[],i);
end
A_inverse = zeros([tsize(2) tsize(1) tsize(3:end)]);
for k = 1 : num_slices
    A_inverse(:,:,k) = pinv(A(:,:,k));
end

for i = p :-1: 3
    A_inverse = ifft(A_inverse,[],i); 
end
return 