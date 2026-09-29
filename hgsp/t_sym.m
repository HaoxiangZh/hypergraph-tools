function As = t_sym(A)
% T_SYM  Symmetrize a tensor along its last mode(s) (t-HGSP).
%   An N x N x N tensor becomes N x N x (2N+1) with frontal slices
%   [0, A(:,:,1:N)/2, A(:,:,N:-1:1)/2]; applied recursively for p > 3.
A_size = size(A);
if A_size(end)==1
    A_size = A_size(1:end-1);
end
As_size = A_size(1:end-1);
N = A_size(2);
As_size(1) = N;
a = [];
str = "";
for i = 1:length(A_size)-1
    str = str+":,";
end
str = str+"i";
if length(A_size)>3
    As_size(3:end) = 2*N+1;
    As = zeros(As_size);
    for i = 1:A_size(end)
        eval("As=[As;0.5*t_sym(A("+str+"))];")
    end
    for i = A_size(end):-1:1
        eval("As=[As;0.5*t_sym(A("+str+"))];")
    end
else
    As = zeros(As_size);
    for i = 1:A_size(end)
        eval("As=[As;0.5*A("+str+")];")
    end
    for i = A_size(end):-1:1
        eval("As=[As;0.5*A("+str+")];")
    end
end
As = fold(As,2*N+1);
return     
