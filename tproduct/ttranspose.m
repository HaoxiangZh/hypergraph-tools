function A = ttranspose(A)
% TTRANSPOSE  Tensor transpose under the t-product.
%   Transposes every frontal slice and reverses the order of slices 2..n
%   along the last mode (recursively for p > 3).
if ~isa(A,'double')
    A = double(A);
end
if(numel(size(A))==2)
    A = A';
else
    A_t = [];
    size_list = size(A);
    str = "";
    for i = 1:length(size_list)-1
        str = str+":,";
    end
    eval("A_t=ttranspose(A("+str+"1));")
    str = str+"i";
    for i = size_list(end):-1:2
        eval("A_t=[A_t;ttranspose(A("+str+"))];")
    end
    A = fold(A_t,size_list(end));
end
