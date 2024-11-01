clc;
clear;
close all;
 %% Data Loading:
data = readtable('classification_results.xlsx','Sheet' ,'HC_DF_SPT_Head and Neck'); 

y_true = data.y_true;
y_pred = data.y_pred;
y_pred_prob = data.y_pred_proba;

%% Confusion Matrix:
conf_mat = confusionmat(y_true, y_pred);
confusionchart(y_true, y_pred);

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


p0 = sum(diag(conf_mat)) / sum(conf_mat(:));
pe = (((TN + FP)* (TN + FN))+ ((TP + FP)*(TP + FN))) / (sum(conf_mat(:))^2);
kappa = (p0 - pe) / (1 - pe);
disp(['Cohen''s Kappa: ', num2str(kappa, '%.15f')]);


MCC = (TP * TN - FP * FN) ./ sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN));
disp(['MCC: ', num2str(MCC, '%.15f')]);

log_loss = -mean(y_true(:) .* log(y_pred_prob(:)) + (1 - y_true(:)) .* log(1 - y_pred_prob(:)));
disp(['Log Loss: ', num2str(log_loss, '%.15f')]);


specificity = TN / (TN + FP);
balanced_accuracy = (recall + specificity)/2;
disp(['Balanced Accuracy: ', num2str(balanced_accuracy, '%.15f')]);


beta = 0.5;
f_beta_score = ((1 + beta^2) * (precision * recall)) / (beta.^2 * precision + recall);
disp(['F-Beta Score: ', num2str(f_beta_score , '%.15f')]);


jaccard_index = jaccard(y_true, y_pred);
disp(['Jaccard Index: ', num2str(jaccard_index , '%.15f')]);


geometric_mean = geomean([recall, specificity]);
disp(['Geometric Mean: ', num2str(mean(geometric_mean) , '%.15f')]);

standard_deviation = std(y_pred) - std(y_true);
disp(['Standard Deviation: ', num2str(mean(standard_deviation) , '%.15f')]);


[X, Y, T, AUC] = perfcurve(y_true, y_pred, 1);
disp(['AUC: ', num2str(mean(AUC) , '%.15f')]);