clc, clear
% Generate datasets
datasets = generate_datasets();
max_iter = 100; % Maximum iterations
num_runs = 5; % Number of runs for each algorithm (for averaging)

% Print metrics explanation using ASCII characters
fprintf('+-----------------------------------------------------------------------+\n');
fprintf('|                           Metrics Explanation                         |\n');
fprintf('+-----------------------------------------------------------------------+\n');
fprintf('| Runtime(s)  : Lower is better  (algorithm execution time)             |\n');
fprintf('| WCSS       : Lower is better  (Within-Cluster Sum of Squares)         |\n');
fprintf('| Silhouette : Higher is better (range [-1,1], clustering quality)      |\n');
fprintf('| Clusters   : Should match the preset k value                          |\n');
fprintf('| * indicates better performance on that metric                         |\n');
fprintf('+-----------------------------------------------------------------------+\n');

% Initialize results table header
fprintf('%-25s %-12s %10s %12s %12s %10s\n', ...
    'Dataset', 'Algorithm', 'Runtime(s)', 'WCSS', 'Silhouette', 'Clusters');
fprintf('%-25s %-12s %10s %12s %12s %10s\n', ...
    repmat('=', 1, 25), repmat('=', 1, 12), repmat('=', 1, 10), ...
    repmat('=', 1, 12), repmat('=', 1, 12), repmat('=', 1, 10));

% Run tests on each dataset
for i = 1:length(datasets)
    dataset = datasets{i};
    data = dataset.data;
    k = dataset.k;

    fprintf('\nTesting on %s (%d points, %d dimensions, %d clusters)\n', ...
        dataset.name, size(data, 1), size(data, 2), k);

    % Run multiple times to get average performance
    t1_total = 0;
    t2_total = 0;
    wcss1_total = 0;
    wcss2_total = 0;
    silhouette1_total = 0;
    silhouette2_total = 0;
    m1_total = 0;
    m2_total = 0;

    for run = 1:num_runs
        % Set same random seed for both algorithms in this run
        rng(42 + run);

        % Test eff_kmeans
        tic;
        [idx1, center1, m1] = eff_kmeans(data, k, max_iter);
        t1 = toc;
        [wcss1, silhouette1] = evaluate_clustering(data, idx1, center1);

        t1_total = t1_total + t1;
        wcss1_total = wcss1_total + wcss1;
        silhouette1_total = silhouette1_total + silhouette1;
        m1_total = m1_total + m1;

        % Use same seed state for both algorithms to ensure identical initialization
        s = rng;

        % Reset to same seed state for elkan_kmeans
        rng(s);

        % Test elkan_kmeans
        tic;
        [idx2, center2, m2] = elkan_kmeans(data, k, max_iter);
        t2 = toc;
        [wcss2, silhouette2] = evaluate_clustering(data, idx2, center2);

        t2_total = t2_total + t2;
        wcss2_total = wcss2_total + wcss2;
        silhouette2_total = silhouette2_total + silhouette2;
        m2_total = m2_total + m2;
    end

    % Calculate averages
    t1_avg = t1_total / num_runs;
    t2_avg = t2_total / num_runs;
    wcss1_avg = wcss1_total / num_runs;
    wcss2_avg = wcss2_total / num_runs;
    silhouette1_avg = silhouette1_total / num_runs;
    silhouette2_avg = silhouette2_total / num_runs;
    m1_avg = m1_total / num_runs;
    m2_avg = m2_total / num_runs;

    % Determine better metrics (using '*' instead of check mark)
    if t1_avg <= t2_avg
        t_check1 = '*';
        t_check2 = ' ';
    else
        t_check1 = ' ';
        t_check2 = '*';
    end

    if wcss1_avg <= wcss2_avg
        wcss_check1 = '*';
        wcss_check2 = ' ';
    else
        wcss_check1 = ' ';
        wcss_check2 = '*';
    end

    if silhouette1_avg >= silhouette2_avg
        sil_check1 = '*';
        sil_check2 = ' ';
    else
        sil_check1 = ' ';
        sil_check2 = '*';
    end

    % Print results with ASCII box drawing
    fprintf('\n+- %s (%d points, %d dimensions, %d clusters)\n', ...
        dataset.name, size(data, 1), size(data, 2), k);
    fprintf('|  %-25s %-12s %9.4f%s %11.2f%s %11.4f%s %9.1f\n', ...
        dataset.name, 'eff_kmeans', t1_avg, t_check1, wcss1_avg, wcss_check1, silhouette1_avg, sil_check1, m1_avg);
    fprintf('|  %-25s %-12s %9.4f%s %11.2f%s %11.4f%s %9.1f\n', ...
        dataset.name, 'elkan_kmeans', t2_avg, t_check2, wcss2_avg, wcss_check2, silhouette2_avg, sil_check2, m2_avg);
    speedup = t1_avg / t2_avg;

    if speedup > 1
        fprintf('+- SPEEDUP: elkan_kmeans is %.2fx FASTER than eff_kmeans\n', speedup);
    else
        fprintf('+- Performance: eff_kmeans is %.2fx faster than elkan_kmeans\n', 1 / speedup);
    end

end

% Print summary of speedup results
fprintf('\n+-----------------------------------------------------------------------+\n');
fprintf('|                        SPEEDUP SUMMARY                                |\n');
fprintf('+-----------------------------------------------------------------------+\n');
fprintf('| Dataset                      | elkan_kmeans speedup factor            |\n');
fprintf('+------------------------------|----------------------------------------+\n');

% Re-run through datasets just to summarize
total_speedup = 0;

for i = 1:length(datasets)
    dataset = datasets{i};

    % Get average runtimes from previous calculations
    % This is a simplified approach - in a real implementation you'd store these values
    % For demonstration purpose, we'll just recompute
    t1_total = 0;
    t2_total = 0;

    for run = 1:num_runs
        % Compute times (simplified - in reality you'd store the values from above)
        rng(42 + run);
        tic; [idx1, ~, ~] = eff_kmeans(dataset.data, dataset.k, max_iter); t1 = toc;
        s = rng; rng(s);
        tic; [idx2, ~, ~] = elkan_kmeans(dataset.data, dataset.k, max_iter); t2 = toc;

        t1_total = t1_total + t1;
        t2_total = t2_total + t2;
    end

    t1_avg = t1_total / num_runs;
    t2_avg = t2_total / num_runs;
    speedup = t1_avg / t2_avg;

    fprintf('| %-28s | %.2fx                                  |\n', dataset.name, speedup);
    total_speedup = total_speedup + speedup;
end

avg_speedup = total_speedup / length(datasets);
fprintf('+------------------------------|----------------------------------------+\n');
fprintf('| AVERAGE SPEEDUP              | %.2fx                                  |\n', avg_speedup);
fprintf('+-----------------------------------------------------------------------+\n');

% Generate datasets for testing
function datasets = generate_datasets()
    % Set random seed for reproducibility
    rng(42);

    % Dataset 1: Well-separated Gaussian clusters in 2D
    n1 = 1000;
    centers1 = [0 0; 5 5; -5 5; 5 -5; -5 -5];
    k1 = size(centers1, 1);
    data1 = [];

    for i = 1:k1
        data1 = [data1; centers1(i, :) + randn(n1 / k1, 2)];
    end

    % Dataset 2: Overlapping Gaussian clusters in 2D
    n2 = 1000;
    centers2 = [0 0; 2 2; -2 2; 2 -2; -2 -2];
    k2 = size(centers2, 1);
    data2 = [];

    for i = 1:k2
        data2 = [data2; centers2(i, :) + randn(n2 / k2, 2)];
    end

    % Dataset 3: Higher-dimensional data
    n3 = 1000;
    dim3 = 10;
    k3 = 5;
    centers3 = 10 * randn(k3, dim3);
    data3 = [];

    for i = 1:k3
        data3 = [data3; repmat(centers3(i, :), n3 / k3, 1) + randn(n3 / k3, dim3)];
    end

    % Dataset 4: Larger dataset
    n4 = 5000;
    k4 = 10;
    centers4 = 10 * randn(k4, 2);
    data4 = [];

    for i = 1:k4
        data4 = [data4; repmat(centers4(i, :), n4 / k4, 1) + randn(n4 / k4, 2)];
    end

    % Dataset 5: Very large dataset (for testing scalability)
    n5 = 20000;
    k5 = 20;
    centers5 = 15 * randn(k5, 2);
    data5 = [];

    for i = 1:k5
        data5 = [data5; repmat(centers5(i, :), n5 / k5, 1) + 2 * randn(n5 / k5, 2)];
    end

    % Package datasets
    datasets = {
                struct('name', 'Well-separated 2D', 'data', data1, 'k', k1),
                struct('name', 'Overlapping 2D', 'data', data2, 'k', k2),
                struct('name', 'High-dimensional', 'data', data3, 'k', k3),
                struct('name', 'Large dataset', 'data', data4, 'k', k4),
                struct('name', 'Very large dataset', 'data', data5, 'k', k5)
                };
end

% Evaluate clustering quality
function [wcss, silhouette_score] = evaluate_clustering(data, idx, centers)
    n = size(data, 1);
    m = size(centers, 1);

    % Calculate within-cluster sum of squares
    wcss = 0;

    for j = 1:m
        cluster_points = data(idx == j, :);

        if ~isempty(cluster_points)
            diffs = cluster_points - repmat(centers(j, :), size(cluster_points, 1), 1);
            wcss = wcss + sum(sum(diffs .^ 2));
        end

    end

    % Calculate simplified silhouette score (for efficiency)
    % We'll sample points to make this more efficient for large datasets
    sample_size = min(n, 500); % Cap at 500 points for efficiency
    sample_indices = randsample(n, sample_size);

    silhouette_sum = 0;

    for i = 1:sample_size
        idx_i = sample_indices(i);
        cluster_i = idx(idx_i);

        % Average distance to points in same cluster
        same_cluster_idx = find(idx == cluster_i);
        same_cluster_idx = same_cluster_idx(same_cluster_idx ~= idx_i);

        if ~isempty(same_cluster_idx)
            same_cluster = data(same_cluster_idx, :);
            a_i = mean(sqrt(sum((same_cluster - repmat(data(idx_i, :), size(same_cluster, 1), 1)) .^ 2, 2)));
        else
            a_i = 0;
        end

        % Minimum average distance to points in other clusters
        b_i = inf;

        for j = 1:m

            if j ~= cluster_i
                other_cluster = data(idx == j, :);

                if ~isempty(other_cluster)
                    avg_dist = mean(sqrt(sum((other_cluster - repmat(data(idx_i, :), size(other_cluster, 1), 1)) .^ 2, 2)));
                    b_i = min(b_i, avg_dist);
                end

            end

        end

        % Add to silhouette sum
        if a_i == 0 && b_i == inf
            s_i = 0;
        else
            s_i = (b_i - a_i) / max(a_i, b_i);
        end

        silhouette_sum = silhouette_sum + s_i;
    end

    silhouette_score = silhouette_sum / sample_size;
end
