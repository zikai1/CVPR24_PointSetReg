function [idx, centers, m] = elkan_kmeans(data, m, MaxIter)
    % elkan_kmeans - Vectorized implementation of k-Means algorithm based on Elkan's acceleration approach
    %
    % Syntax:
    %   [idx, centers, m] = elkan_kmeans(data, m, MaxIter)
    %
    % Inputs:
    %   data    - n x d data matrix, one sample per row
    %   m       - Initial number of clusters
    %   MaxIter - Maximum number of iterations
    %
    % Outputs:
    %   idx     - n x 1 cluster assignment vector (values from 1 to m)
    %   centers - m x d matrix of cluster centers
    %   m       - Actual number of clusters (may decrease after removing empty clusters)
    %
    % Notes:
    %   This algorithm leverages an upper bound u(i) (distance from each point to its assigned center)
    %   and a lower bound matrix L to avoid unnecessary distance calculations. Using distances between
    %   centers, it computes a "protection radius" s. For data points where u(i) <= s(idx(i)),
    %   distances to other centers need not be computed.
    %   When updating centers, empty clusters are removed. The same removal is applied to old centers
    %   to ensure array sizes match when calculating center movement distances, thus avoiding
    %   dimension mismatch errors.

    [n, d] = size(data);
    % Randomly initialize centers (select m samples)
    perm = randperm(n);
    centers = data(perm(1:m), :);

    % Initially compute distances from all points to all centers (using sqdist; note data points must be column-wise)
    D = sqdist(centers', data');
    D = sqrt(D); % Euclidean distance matrix, size m x n
    % Assign each point to its nearest center
    [u, idx] = min(D, [], 1);
    u = u';
    idx = idx';
    % Lower bound matrix: n x m, each row stores distances from a point to all centers
    L = D';

    % Calculate distances between centers and determine protection radius s(j)=0.5*min_{l~=j} d(centers(j),centers(l))
    centerD = sqdist(centers', centers');
    centerD = sqrt(centerD);
    s = zeros(m, 1);

    for j = 1:m
        temp = centerD(j, :);
        temp(j) = inf;
        s(j) = 0.5 * min(temp);
    end

    for iter = 1:MaxIter
        % Step 1: Use upper bounds and protection radii to identify points that need recalculation of all center distances
        recheck = find(u > s(idx));

        if ~isempty(recheck)
            % For points in recheck, vectorize calculation of distances to all centers
            D_recheck = sqdist(centers', data(recheck, :)');
            D_recheck = sqrt(D_recheck);
            % Update lower bound matrix
            L(recheck, :) = D_recheck';
            % Update upper bounds and center assignments
            [new_u, new_idx] = min(D_recheck, [], 1);
            new_u = new_u';
            new_idx = new_idx';
            u(recheck) = new_u;
            idx(recheck) = new_idx;
        end

        % Step 2: Update cluster centers
        % Save current centers for later calculation of center movement distances
        oldCenters = centers;
        newCenters = zeros(m, d);

        for j = 1:m
            members = find(idx == j);

            if isempty(members)
                newCenters(j, :) = centers(j, :); % Maintain original center for empty clusters
            else
                newCenters(j, :) = mean(data(members, :), 1);
            end

        end

        % Remove empty clusters: if a cluster has no points, remove it and synchronously update idx, old centers,
        % lower bound matrix L and protection radius s
        activeClusters = false(m, 1);

        for j = 1:m

            if any(idx == j)
                activeClusters(j) = true;
            end

        end

        if sum(activeClusters) < m
            newMapping = cumsum(activeClusters);
            idx = newMapping(idx);
            newCenters = newCenters(activeClusters, :);
            oldCenters = oldCenters(activeClusters, :); % Synchronously remove empty clusters
            L = L(:, activeClusters);
            s = s(activeClusters);
            m = sum(activeClusters);
        end

        % Step 3: Calculate center movement distances and update upper and lower bounds
        center_shift = sqrt(sum((oldCenters - newCenters) .^ 2, 2));
        centers = newCenters;

        % Update upper bound for each point: add the movement of its assigned center
        u = u + center_shift(idx);
        % Update lower bounds: for each center j, reduce corresponding lower bound by center_shift(j), ensuring non-negative
        L = max(L - repmat(center_shift', n, 1), 0);

        % Update distances between centers and protection radii s
        centerD = sqdist(centers', centers');
        centerD = sqrt(centerD);

        for j = 1:m
            temp = centerD(j, :);
            temp(j) = inf;
            s(j) = 0.5 * min(temp);
        end

        % Consider algorithm converged if all center movements are very small
        if max(center_shift) < 1e-6
            break;
        end

    end

end
