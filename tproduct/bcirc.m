function a = bcirc(A)
% BCIRC  Block circulant matrix of a p-order tensor along its last mode.
%   a = bcirc(A). For an n1 x n2 x n3 tensor the result is (n1*n3) x (n2*n3).
%   For p > 3 the blocks are the (p-1)-order slices of A.


size_list = size(A);
size_end = size_list(end);
a = [];
str = "";
for i = 1:length(size_list)-1
    str = str+":,";
end
for i = 1:size_end
    aa = [];
    for j = 1 : size_end
        t = i-j+1;
        if t<=0
            t = t+size_end;
        end
        eval("aa = [aa A("+str+"t)];")
    end
    a = [a;aa];
end
return 