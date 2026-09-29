function A_fold = fold(A,p)
% FOLD  Inverse of UNFOLD.
%   A_fold = fold(A, p) splits A into p blocks along its first mode and
%   stacks them along a new last mode.

A_size = size(A);
A_fold_size = [A_size(1)/p A_size(2:end) p];
A_fold = zeros(A_fold_size);
str = "";
for i = 1:length(A_size)-1
    str = str+",:";
end
n = A_size(1)/p;
for i = 1:p
    eval("A_fold(:"+str+",i) = A((i-1)*n+1:i*n"+str+");")
end
return 