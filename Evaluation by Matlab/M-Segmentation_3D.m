clc;
clear;
close all
%% clc;
clear;
close all;

%% Data Loading:
% Load 3D data from .mat file
data = load('random_3d_data.mat');
y_true = data.y_true;
y_pred = data.y_pred;

% Ensure y_true and y_pred are of size (64, 64, 64)
assert(all(size(y_true) == [64, 64, 64]), 'y_true must be of size (64, 64, 64)');
assert(all(size(y_pred) == [64, 64, 64]), 'y_pred must be of size (64, 64, 64)');

% Convert to double for calculations if needed
y_true = double(y_true);
y_pred = double(y_pred);

% Convert 3D arrays to 1D for confusion matrix calculation
y_true_1D = y_true(:);
y_pred_1D = y_pred(:);

%% Confusion Matrix:
confusionchart(y_true_1D, y_pred_1D);
conf_mat = confusionmat(y_true_1D, y_pred_1D);
n = size(conf_mat, 1);
TP = conf_mat(2, 2);
FP = conf_mat(1, 2);
FN = conf_mat(2, 1);
TN = conf_mat(1, 1);

%% Calculating Evaluation Metrics:

accuracy = sum(diag(conf_mat)) / sum(conf_mat(:));
disp(['Accuracy: ', num2str(accuracy, '%.15f')]);

precision = TP / (TP + FP);
disp(['Precision: ', num2str(precision, '%.15f')]);

recall = TP / (TP + FN);
disp(['Recall: ', num2str(recall, '%.15f')]);

F1_score = 2 * (precision * recall) / (precision + recall);
disp(['F1 Score: ', num2str(F1_score, '%.15f')]);

dice_index = dice(y_true_1D, y_pred_1D);
disp(['Dice Index = ' num2str(dice_index, '%.15f')]);

Jaccard = jaccard(y_true_1D, y_pred_1D);
disp(['Jaccard Index = ' num2str(Jaccard, '%.15f')]);

bf_score = bfscore(y_true_1D, y_pred_1D);
disp(['BF Score = ' num2str(bf_score, '%.15f')]);

IoU_foreground = TP / (TP + FP + FN);
IoU_background = TN / (TN + FP + FN);
IOU = mean(Jaccard); % Assuming Jaccard is already calculated as a mean value
disp(['IoU = ' num2str(IOU, '%.15f')]);

IOU_mean = (IoU_foreground + IoU_background) / 2;
disp(['Mean IoU = ' num2str(IOU_mean, '%.15f')]);

% Hausdorff Distance
bound_true = bwperim(y_true_1D);
bound_pred = bwperim(y_pred_1D);
dist_true = bwdist(bound_true);
dist_pred = bwdist(bound_pred);
max_dist_true_to_pred = max(dist_pred(bound_true));
max_dist_pred_to_true = max(dist_true(bound_pred));
hausdorff_distance = max(max_dist_true_to_pred, max_dist_pred_to_true);
disp(['HausDorff = ' num2str(hausdorff_distance, '%.15f')]);

% Cohen's Kappa
p0 = sum(diag(conf_mat)) / sum(conf_mat(:));
pe = (((TN + FP) * (TN + FN)) + ((TP + FP) * (TP + FN))) / (sum(conf_mat(:))^2);
kappa = (p0 - pe) / (1 - pe);
disp(['Cohen''s Kappa: ', num2str(kappa, '%.15f')]);

specificity = TN / (TN + FP);
geometric_mean = geomean([recall, specificity]);
disp(['Geometric Mean: ', num2str(geometric_mean, '%.15f')]);
