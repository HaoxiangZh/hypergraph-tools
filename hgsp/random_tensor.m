function A = random_tensor(tsize)
% RANDOM_TENSOR  Random tensor of size tsize with symmetric frontal slices,
%   symmetrized by T_SYM.
num_slices = prod(tsize(3:end));
A = zeros([tsize(1:2), num_slices]);
for i = 1:num_slices
    A(:,:,i) = rand(tsize(1:2));
    A(:,:,i) = A(:,:,i)+A(:,:,i)';
end
A = reshape(A,tsize);
A = t_sym(A);