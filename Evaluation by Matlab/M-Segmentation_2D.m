clc;
clear;
close all;
%% Data Loading:
% Data 1
data = readtable('binary_segmentation.xlsx', 'Sheet', 'Sheet1');
y_true = reshape(data.y_true, [128, 128]);
y_pred = reshape(data.y_pred, [128, 128]);

y_true_1D = y_true(:);
y_pred_1D = y_pred(:);
% 
% % Data 2:
% y_true = [[0, 1, 1]; [0, 0, 1]; [1, 1, 0]];
% y_pred = [[0, 1, 1]; [0, 1, 1]; [1, 0, 0]];
% 
% y_true_1D = y_true(:);
% y_pred_1D = y_pred(:);

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

precision = TP/(FP + TP);
disp(['Precision: ', num2str(precision, '%.15f')]);


recall = TP / (TP + FN);
disp(['Recall: ', num2str(mean(recall), '%.15f')]);

F1_score = 2 * (precision * recall) / (precision + recall);
disp(['F1 Score: ', num2str(F1_score, '%.15f')]);

dice_index = dice(y_true, y_pred);
disp(['Dice Index = ' num2str(dice_index, '%.15f')]);

Jaccard = jaccard(y_true, y_pred);
disp(['Jaccard Index = ' num2str(Jaccard, '%.15f')]);

bf_score = bfscore(y_true, y_pred);
disp(['bf_score = ' num2str(bf_score, '%.15f')]);


IoU_foreground = TP / (TP + FP + FN);
IoU_background = TN / (TN + FP + FN);
%IOU_mean =  (IoU_background + IoU_foreground) / 2;
IOU= mean(Jaccard);
disp(['IoU = ' num2str(IOU, '%.15f')]);

IOU_mean = (IoU_foreground + IoU_background)/2;
disp(['Mean IoU = ' num2str(IOU_mean, '%.15f')]);


bound_true = bwperim(y_true);
bound_pred = bwperim(y_pred);
dist_true = bwdist(bound_true);
dist_pred = bwdist(bound_pred);
max_dist_true_to_pred = max(dist_pred(bound_true));
max_dist_pred_to_true = max(dist_true(bound_pred));
hausdorff_distance = max(max_dist_true_to_pred, max_dist_pred_to_true);
disp(['HausDorff = ' num2str(hausdorff_distance, '%.15f')]);



p0 = sum(diag(conf_mat)) / sum(conf_mat(:));
pe = (((TN + FP)* (TN + FN))+ ((TP + FP)*(TP + FN))) / (sum(conf_mat(:))^2);
kappa = (p0 - pe) / (1 - pe);
disp(['Cohen''s Kappa: ', num2str(kappa, '%.15f')]);



specificity = TN / (TN + FP);
geometric_mean = geomean([recall, specificity]);
disp(['Geometric Mean: ', num2str(mean(geometric_mean) , '%.15f')]);


