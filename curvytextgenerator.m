function curvytextgenerator(xy, str, varargin)
    if size(xy,1) > 2
        xy = xy';
    end

    n = size(xy,2);
    m = length(str);
    
    % Calculate the spline
    XY = spline(1:n, xy, linspace(1,n,m+1));
    
    % Calculate the distances between adjacent points on the curve
    dXY = sqrt(sum((XY(:,2:end) - XY(:,1:end-1)).^2));
    
    % Compute the cumulative sum of distances
    cumulativeDist = [0 cumsum(dXY)];
    
    % Calculate the spacing for each letter
    letterSpacing = cumulativeDist(end) / m;
    
    letterIndex = 1;
    
    for i = 1:m
        letterIndex = min(letterIndex, m - 1);
        while cumulativeDist(letterIndex) < (i - 1) * letterSpacing && letterIndex < m
            letterIndex = letterIndex + 1;
        end

        letterPos = interp1(cumulativeDist, XY', (i - 0.5) * letterSpacing);
        letterPos = letterPos';

        dXY_temp = XY(:, letterIndex + 1) - XY(:, letterIndex);
        theta = atan2(dXY_temp(2), dXY_temp(1)) / (2 * pi) * 360;

        text(letterPos(1), letterPos(2), str(i), 'rotation', theta, ...
             'horizontalalignment', 'center', 'verticalalignment', 'middle', varargin{:});
    end
    
    minX = min(xy(1,:));
    minY = min(xy(2,:));
    
    maxX = max(xy(1,:));
    maxY = max(xy(2,:));
    
    %xlim([min([minX, minY]), max([maxX, maxY])]);
    %ylim([min([minX, minY]), max([maxX, maxY])]);
    
    xlim([min(xy(1,:)), max(xy(1,:))]);
    ylim([min(xy(2,:)), max(xy(2,:))]);
    
    axis equal;
    axis off;
end
