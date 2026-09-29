function visualization_t_product(C,A,B,coordinateInfo)
% VISUALIZATION_T_PRODUCT  Interactive 3-D view of C = A * B.
%   Clicking an entry of C highlights the entries of A and B that contribute
%   to it (coordinateInfo from T_PRODUCT_ENTRYWISE).
    szC = size(C);
    szA = size(A);
    szB = size(B);
    XYZ = { ...
      [0 0 0 0]  [0 0 1 1]  [0 1 1 0] ; ...
      [1 1 1 1]  [0 0 1 1]  [0 1 1 0] ; ...
      [0 1 1 0]  [0 0 0 0]  [0 0 1 1] ; ...
      [0 1 1 0]  [1 1 1 1]  [0 0 1 1] ; ...
      [0 1 1 0]  [0 0 1 1]  [0 0 0 0] ; ...
      [0 1 1 0]  [0 0 1 1]  [1 1 1 1]   ...
    };

    xcube = cat(1, XYZ{:,1})';
    ycube = cat(1, XYZ{:,2})';
    zcube = cat(1, XYZ{:,3})';

    [x0, y0, z0] = ndgrid(1:szC(1), 1:szC(2), 1:szC(3));
    [x1, y1, z1] = ndgrid(1:szA(1), 1:szA(2), 1:szA(3));
    [x2, y2, z2] = ndgrid(1:szB(1), 1:szB(2), 1:szB(3));
    colsC = C(:);
    colsA = A(:);
    colsB = B(:);

    xallC = arrayfun(@(a) a + xcube, x0(:), 'uni', 0);
    yallC = arrayfun(@(a) a + ycube, y0(:), 'uni', 0);
    zallC = arrayfun(@(a) a + zcube, z0(:), 'uni', 0);
    xallA = arrayfun(@(a) a + xcube, x1(:), 'uni', 0);
    yallA = arrayfun(@(a) a + ycube, y1(:), 'uni', 0);
    zallA = arrayfun(@(a) a + zcube, z1(:), 'uni', 0);
    xallB = arrayfun(@(a) a + xcube, x2(:), 'uni', 0);
    yallB = arrayfun(@(a) a + ycube, y2(:), 'uni', 0);
    zallB = arrayfun(@(a) a + zcube, z2(:), 'uni', 0);
    xallC = cat(2, xallC{:});
    yallC = cat(2, yallC{:});
    zallC = cat(2, zallC{:});
    xallA = cat(2, xallA{:});
    yallA = cat(2, yallA{:});
    zallA = cat(2, zallA{:});
    xallB = cat(2, xallB{:});
    yallB = cat(2, yallB{:});
    zallB = cat(2, zallB{:});
    colsC = kron(colsC(:)', ones(1, 6));
    colsC = colsC / max(colsC);
    colsA = kron(colsA(:)', ones(1, 6));
    colsA = colsA / max(colsA);
    colsB = kron(colsB(:)', ones(1, 6));
    colsB = colsB / max(colsB);
    figure;
    Cx = subplot(1,3,1);
    title(Cx, 'C');
    pC = patch(xallC, yallC, zallC, colsC, 'EdgeColor', 'none', 'FaceAlpha', 0.3, 'DisplayName', 'main');
    pC.Tag = 'C';
    pC.AlphaDataMapping = 'none';
    colormap(parula);
    ylim(Cx,[0, szC(2)]);
    axis equal;
    view([45 25]);
    xlabel('X');
    ylabel('Y');
    zlabel('Z');

    Ax = subplot(1,3,2);
    title(Ax, 'A');
    pA = patch(xallA, yallA, zallA, colsA, 'EdgeColor', 'none', 'FaceAlpha', 0.3, 'DisplayName', 'main');
    pA.Tag = 'A';
    pA.AlphaDataMapping = 'none';
    colormap(parula);
    ylim(Ax,[0, szA(2)]);
    axis equal;
    view([45 25]);
    xlabel('X');
    ylabel('Y');
    zlabel('Z');

    Bx = subplot(1,3,3);
    title(Bx, 'B');
    pB = patch(xallB, yallB, zallB, colsB, 'EdgeColor', 'none', 'FaceAlpha', 0.3, 'DisplayName', 'main');
    pB.Tag = 'B';
    pB.AlphaDataMapping = 'none';
    colormap(parula);
    ylim(Bx,[0, szB(2)]);
    axis equal;
    view([45 25]);
    xlabel('X');
    ylabel('Y');
    zlabel('Z');


    
    % 创建数据光标模式
    dcm_obj = datacursormode(gcf);
    set(dcm_obj, 'DisplayStyle', 'datatip', 'SnapToDataVertex', 'on', 'UpdateFcn', {@tooltipFunction, szC, coordinateInfo,pC,pA,pB});
    
    % Create buttons to collect X/Y/Z indices
    indexButton = uicontrol('Style', 'pushbutton', 'String', 'Index', 'Position', [10 20 80 30], 'Callback', @getIndex);
    
    function pos = getIndex(~, ~)
        indices = str2double(inputdlg({'Enter X Index:', 'Enter Y Index:', 'Enter Z Index:'}, 'Input', 1, {'0', '0', '0'}));
        xIndex = indices(1); % 获取 X 索引
        yIndex = indices(2); % 获取 Y 索引
        zIndex = indices(3); % 获取 Z 索引
        pos = indices';
        highlightRowBlocks(pos);
        % 现在你可以使用 xIndex、yIndex 和 zIndex 进行后续操作
    end

    % 创建一个回调函数，在选中一个小方块时显示其位置和颜色
    function txt = tooltipFunction(~, event_obj, sz, coordinateInfo,pC,pA,pB)
        % patchName = get(patchObj, 'DisplayName');
        pos = get(event_obj, 'Position');
        % display(event_obj.Target.Tag);
        if pos(1) > sz(1)
            pos(1) = pos(1) - 1;
        end
        if pos(2) > sz(2)
            pos(2) = pos(2) - 1;
        end
        if pos(3) > sz(3)
            pos(3) = pos(3) - 1;
        end
        colValue = getColorValue(pos,sz);
        % switch patchName
        if event_obj.Target.Tag == 'C'
            highlightRowBlocks(pos, coordinateInfo,pC,pA,pB);
        end
        txt = {['Column: ' num2str(pos(1))], ['Row: ' num2str(pos(2))], ['Face: ' num2str(pos(3))], ['Color Value: ' num2str(colValue)]};
    
    end

    % 获取选中位置的颜色值
    function colValue = getColorValue(pos,sz)
        cols = round(pos(1));
        row = round(pos(2));
        face = round(pos(3));
        idx = sub2ind(sz, row, cols, face);
        colValue = C(idx);
    end


   % 高亮对应的行方块
    function highlightRowBlocks(pos, coordinateInfo,pC,pA,pB)
        x = round(pos(1));
        y = round(pos(2));
        z = round(pos(3));
        rowsA = [];
        rowsB = [];
        rowsC = find(pC.Vertices(:,1)==x & pC.Vertices(:,2)==y & pC.Vertices(:,3)==z);
        for i = 1:length(coordinateInfo{x,y,z})
            rowsA = [rowsA find(pA.Vertices(:,1) == coordinateInfo{x,y,z}(i).A_coords(1,1)&pA.Vertices(:,2) == coordinateInfo{x,y,z}(i).A_coords(1,2)&pA.Vertices(:,3) == coordinateInfo{x,y,z}(i).A_coords(1,3))'];
            rowsB = [rowsB find(pB.Vertices(:,1) == coordinateInfo{x,y,z}(i).B_coords(1,1)&pB.Vertices(:,2) == coordinateInfo{x,y,z}(i).B_coords(1,2)&pB.Vertices(:,3) == coordinateInfo{x,y,z}(i).B_coords(1,3))'];
        end
        % 更新对应行方块的顶点颜色
        pC.FaceVertexAlphaData = ones(size(pC.Vertices, 1), 1) * 0.1;  % 透明度为0.1
        pC.FaceVertexAlphaData(rowsC,1) = 1;
        pC.FaceAlpha = 'flat';  % 使用 FaceVertexAlphaData 指定的透明度

        pA.FaceVertexAlphaData = zeros(size(pA.Vertices, 1), 1);  % 透明度为0
        pA.FaceVertexAlphaData(rowsA,1) = 1;
        pA.FaceAlpha = 'flat';  % 使用 FaceVertexAlphaData 指定的透明度

        pB.FaceVertexAlphaData = zeros(size(pB.Vertices, 1), 1);  % 透明度为0
        pB.FaceVertexAlphaData(rowsB,1) = 1;
        pB.FaceAlpha = 'flat';  % 使用 FaceVertexAlphaData 指定的透明度
    end



end
