clc;
clear
close all
%% Data Loading:
data = readtable('Processed_LC_RF_PT_Lung.xlsx');
clusters = data.clusters;
number_of_cols = width(data);
X = table2array(data(:, 1:number_of_cols-1));

%%  Calculating Evaluation Metrics:

silhouette_score = silhouette(X, clusters, 'Euclidean');
disp(['Silhouette : ' num2str(mean(silhouette_score), '%.15f')]);

db_index = davies_bouldin_index(X, clusters);
fprintf('Davies-Bouldin Index (MATLAB): %.20f\n', db_index);

ch_index = calinski_harabasz_index(X, clusters);
disp(['ch_index = ' num2str(ch_index, '%.15f')]);


cluster_labels = unique(clusters); 
n_clusters = length(cluster_labels);  
[m, n] = size(X); 
centers = zeros(n_clusters, n);


for i = 1:n_clusters
    cluster_points = X(clusters == cluster_labels(i), :);
    centers(i, :) = mean(cluster_points, 1);
end


wcss = 0;
num_clusters = size(centers, 1);

for k = 1:num_clusters

    cluster_points = X(cluster_labels == k, :);
    distances = sum((cluster_points - centers(k, :)).^2, 2);
    wcss = wcss + sum(distances);
end
disp(['WCSS: ', num2str(wcss, '%.15f')])










%% Functions:

function dbi = davies_bouldin_index(X, labels)
    labels = labels(:);
    clusters = unique(labels);
    numClusters = length(clusters);
    centroids = zeros(numClusters, size(X, 2));
    for k = 1:numClusters
        centroids(k, :) = mean(X(labels == clusters(k), :), 1);
    end
    
    % Compute intra-cluster distances
    intraClusterDistances = zeros(numClusters, 1);
    for k = 1:numClusters
        distances = sqrt(sum((X(labels == clusters(k), :) - centroids(k, :)).^2, 2));
        intraClusterDistances(k) = mean(distances);
    end
    
    % Compute pairwise distances between centroids
    pairwiseDistances = zeros(numClusters);
    for i = 1:numClusters
        for j = i+1:numClusters
            pairwiseDistances(i, j) = sqrt(sum((centroids(i, :) - centroids(j, :)).^2));
            pairwiseDistances(j, i) = pairwiseDistances(i, j); % Symmetric matrix
        end
    end
    
    % Compute the Davies-Bouldin Index
    dbi = 0;
    for i = 1:numClusters
        maxRatio = 0;
        for j = 1:numClusters
            if i ~= j
                ratio = (intraClusterDistances(i) + intraClusterDistances(j)) / pairwiseDistances(i, j);
                if ratio > maxRatio
                    maxRatio = ratio;
                end
            end
        end
        dbi = dbi + maxRatio;
    end
    dbi = dbi / numClusters;
end







function ch_index = calinski_harabasz_index(X, labels)
    % Ensure labels is a column vector
    labels = labels(:);
    
    % Get unique clusters and number of clusters
    clusters = unique(labels);
    numClusters = length(clusters);
    
    % Compute overall centroid
    overall_centroid = mean(X, 1);
    
    % Compute cluster centroids
    centroids = zeros(numClusters, size(X, 2));
    for k = 1:numClusters
        centroids(k, :) = mean(X(labels == clusters(k), :), 1);
    end
    
    % Compute between-cluster dispersion
    between_dispersion = 0;
    for k = 1:numClusters
        cluster_size = sum(labels == clusters(k));
        between_dispersion = between_dispersion + cluster_size * sum((centroids(k, :) - overall_centroid) .^ 2);
    end
    
    % Compute within-cluster dispersion
    within_dispersion = 0;
    for k = 1:numClusters
        cluster_points = X(labels == clusters(k), :);
        distances = sum((cluster_points - centroids(k, :)) .^ 2, 2);
        within_dispersion = within_dispersion + sum(distances);
    end
    
    % Calculate the Calinski-Harabasz Index
    n = size(X, 1); % Total number of data points
    ch_index = (between_dispersion / (numClusters - 1)) / (within_dispersion / (n - numClusters));
end



function fmi = fowlkes_mallows_index(true_labels, cluster_labels)
    % Ensure labels are column vectors
    true_labels = true_labels(:);
    cluster_labels = cluster_labels(:);
    
    % Compute the confusion matrix
    [contingency_table, ~, ~] = crosstab(true_labels, cluster_labels);
    
    % Number of true classes and clusters
    num_true_classes = size(contingency_table, 1);
    num_clusters = size(contingency_table, 2);
    
    % Initialize counts
    TP = 0;
    total_pairs_true = 0;
    total_pairs_pred = 0;
    
    % Calculate True Positives (TP)
    for i = 1:num_true_classes
        for j = 1:num_clusters
            if contingency_table(i, j) > 1
                TP = TP + nchoosek(contingency_table(i, j), 2);
            end
        end
    end
    
    for i = 1:num_true_classes
        if sum(contingency_table(i, :)) > 1
            total_pairs_true = total_pairs_true + nchoosek(sum(contingency_table(i, :)), 2);
        end
    end
    
    % Total possible pairs in the predicted clusters
    for j = 1:num_clusters
        if sum(contingency_table(:, j)) > 1
            total_pairs_pred = total_pairs_pred + nchoosek(sum(contingency_table(:, j)), 2);
        end
    end

    precision = TP / total_pairs_pred;
    recall = TP / total_pairs_true;
    
    % Calculate Fowlkes-Mallows Index
    fmi = sqrt(precision * recall);
end








