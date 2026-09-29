function B = unfold(B)
% UNFOLD  Stack the slices of B along its last mode vertically (block column).

size_list = size(B);
size_end = size_list(end);
b = [];
str = "";
for i = 1:length(size_list)-1
    str = str+":,";
end
str = str+"i";

for i = 1:size_end
    eval("b=[b;B("+str+")];")
end
B = b;
return 