function [C coordinateInfo]= t_product_entrywise(varargin)
% T_PRODUCT_ENTRYWISE  3rd-order t-product computed entry by entry.
%   [C, coordinateInfo] = t_product_entrywise(A, B) also records, for every
%   C(i,j,k), which entries of A and B contribute to it.
%   [C, coordinateInfo] = t_product_entrywise(A, B, coordinateInfo_prev)
%   chains a previous coordinateInfo (e.g. for A * B * C).
%   See also PRINTEQUATIONS, VISUALIZATION_T_PRODUCT.
    
    if nargin < 2
        error('At least two tensors are required as input');
    elseif nargin == 2
        A = varargin{1};
        B = varargin{2};
        [N1, NN, N3] = size(A);
        [~, N2, ~] = size(B);
        coordinateInfo = cell(N1,N2,N3);
        coordinateList = struct('A_coords',[],'B_coords',[]);
        num_coordinates = 0;
    elseif nargin == 3
        A = varargin{1};
        B = varargin{2};
        [N1, NN, N3] = size(A);
        [~, N2, ~] = size(B);
        coordinateInfo_input = varargin{3};
        num_coordinates = length(fieldnames(coordinateInfo_input{1,1,1}));
        coordinateList = struct('A_coords',[],'B_coords',[],'C_coords',[]);
        coordinateInfo = cell(N1,N2,N3);
    else
        error('Too many input arguments');
    end
    Error = 10e-3;
    C = zeros(N1, N2, N3);
    
    for i = 1:N1
        for j = 1:N2
            for k = 1:N3
                for l = 1:N3
                    for m = 1:NN
                        n = mod(k - l, N3)+1;
                        if(A(i,m,n)~=0 & B(m,j,l)~=0)
                            C(i, j, k) = C(i, j, k) + A(i, m, n) * B(m, j, l);
                            % if(C(i,j,k)>error)
                            if num_coordinates == 0
                                coordinateList(end+1).A_coords = [i,m,n,A(i, m, n)];
                                coordinateList(end).B_coords = [m,j,l,B(m, j, l)];
                            elseif num_coordinates == 2
                                coordinateList_input = coordinateInfo_input{i,m,n};
                                for ii = 1:length(coordinateList_input)
                                    coordinateList(end+1).A_coords = coordinateList_input(ii).A_coords;
                                    coordinateList(end).B_coords = coordinateList_input(ii).B_coords;
                                    coordinateList(end).C_coords = [m,j,l,B(m,j,l)];
                                end
                            end
                            % end
    
                        end
                    end
                end
                coordinateInfo{i,j,k} = coordinateList(2:end);
            end
        end
    end
end