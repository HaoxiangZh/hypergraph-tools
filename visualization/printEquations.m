function printEquations(C,coordinateInfo)
% PRINTEQUATIONS  Print the scalar equation behind every entry of C = A * B,
%   using the coordinateInfo returned by T_PRODUCT_ENTRYWISE.
    [N1, N2, N3] = size(coordinateInfo);
    
    for i = 1:N1
        for j = 1:N2
            for k = 1:N3
                equation = [];
                coordinateList = coordinateInfo{i, j, k};
                if ~isempty(coordinateList)
                    for idx = 1:length(coordinateList)
                        A_coords = coordinateList(idx).A_coords;
                        B_coords = coordinateList(idx).B_coords;
                        % C_coords = coordinateList(idx).C_coords;
                        A_str = sprintf('A_%d%d%d(%.3f)', A_coords(1), A_coords(2), A_coords(3), A_coords(4));
                        B_str = sprintf('B_%d%d%d(%.3f)', B_coords(1), B_coords(2), B_coords(3), B_coords(4));
                        % C_str = sprintf('C_%d%d%d(%.3f)', C_coords(1), C_coords(2), C_coords(3), C_coords(4));
                        % term = sprintf('%s * %s *%s', A_str, B_str, C_str);
                        term = sprintf('%s * %s', A_str, B_str);
                        equation = [equation, term];
                        if idx < length(coordinateList)
                            if mod(idx,2)==0
                                equation = [equation, sprintf('\n\t')];
                            end
                            equation = [equation, ' + '];
                        end
                    end
                    fprintf('X_%d%d%d(%.3f) = \n\t\t%s\n', i, j, k, C(i,j,k), equation);
                end
            end
        end
    end
end