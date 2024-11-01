clc;
clear;
close all;
%% Data Loading:

data = readtable('y_true_y_pred_multi.csv'); 
y_true = data.y_true;
y_predict = data.y_pred;

%% Confusion Matrix:
confusionchart(y_true, y_predict);
conf_mat = confusionmat(y_true, y_predict);

%% Calculating Evaluation Metrics:

accuracy = sum(diag(conf_mat)) / sum(conf_mat(:));

n = length(conf_mat);
jaccard = zeros(n, 1);
W = zeros(n, 1);
TP = zeros(n, 1);
TN = zeros(n, 1);
FP = zeros(n, 1);
FN = zeros(n, 1);
overall_j = 0;

for i = 1:n
    TP(i) = conf_mat(i, i); 
    FP(i) = sum(conf_mat(:, i)) - TP(i);  
    FN(i) = sum(conf_mat(i, :)) - TP(i);  
    TN(i) = sum(conf_mat(:))-(TP(i) + FN(i) + FP(i));
    W(i) = sum(conf_mat(i, :))/ sum(conf_mat(:));
    if (TP(i) + FP(i) + FN(i)) > 0
            jaccard(i) = TP(i) / (TP(i) + FP(i) + FN(i));
            overall_j = overall_j + jaccard(i);
        else
            jaccard(i) = 0;
     end

end

precision = TP ./ (TP + FP);
recall = TP ./ (TP + FN);
F1_score = 2*(precision.*recall)./(precision + recall);



macro_precision  = mean(precision);
macro_recall = mean(recall);
macro_jaccard = mean(jaccard);


micro_precision = sum(TP)/sum(TP + FP);
micro_recall = sum(TP)/sum((TP + FN));
micro_F1_score = 2*(micro_precision * micro_recall)/(micro_precision + micro_recall);
if sum(TP + FP + FN) > 0
            maicro_jaccard = sum(TP) / sum(TP + FP + FN);
        else
            maicro_jaccard = 0;
end


weighted_recall = sum(recall .* W);
weighted_F1_score = sum(F1_score.*W);
weighted_precision = sum(precision.*W);
weighted_Jaccard = sum(jaccard.*W);


c = sum(diag(conf_mat));
s = sum(conf_mat(:));
kappa = (c*s - sum((FN + TP) .* (FP + TP))) / (s^2 - sum((FN + FP) .* (FP + FN)));
p0 = c/sum(conf_mat(:));
pe = sum((FN + TP) .* (FP + TP)/ s^2 );
kappa1 = (p0 - pe) / (1 - pe);


K = size(conf_mat, 1);    
pk = sum(conf_mat, 1);   
tk = sum(conf_mat, 2);    
numerator = (c * s) - sum(pk .* tk');
denominator = sqrt((s^2 - sum(pk.^2)) * (s^2 - sum(tk'.^2)));
MCC = numerator / denominator;
balanced_accuracy = sum(TP./(FN + TP)) /(n*sum(W));




beta = 0.5;
f_beta_score = ((1 + beta^2) .* TP) ./ (((1 + beta^2).*TP) + (beta^2 .* FN) + FP);
marco_f_beta_score = mean(f_beta_score);
mirco_f_beta_score = (1 + beta^2) *(micro_precision * micro_recall) / (((beta^2 * micro_precision) + micro_recall));
weighted_f_beta_score = sum(f_beta_score.*W);



precision
disp(['Precision std: ', num2str(std(precision), '%.15f')]);
% recall
disp(['Recall std: ', num2str(std(recall), '%.15f')]);
F1_score
disp(['F1_score std: ', num2str(std(F1_score), '%.15f')]);
jaccard
disp(['jaccard std: ', num2str(std(jaccard), '%.15f')]);

disp(['Accuracy: ', num2str(accuracy, '%.15f')]);

disp(['Macro Precision: ', num2str(macro_precision, '%.15f')]);
disp(['Micro Precision: ', num2str(micro_precision, '%.15f')]);
disp(['Weighted Precision: ', num2str(weighted_precision, '%.15f')]);


disp(['Macro Recall: ', num2str(macro_recall, '%.15f')]);
disp(['Micro Recall: ', num2str(micro_recall, '%.15f')]);
disp(['Weighted Recall: ', num2str(weighted_recall, '%.15f')]);

disp(['Macro_F1_score: ', num2str(mean(F1_score), '%.15f')]);
disp(['Micro_F1_score: ', num2str(micro_F1_score, '%.15f')]);
disp(['Weighted_F1_score: ', num2str(weighted_F1_score, '%.15f')]);


disp(['Cohen''s Kappa: ', num2str(kappa1, '%.15f')]);

disp(['MCC: ', num2str(mean(MCC), '%.15f')]);

disp(['Weighted Balanced Accuracy: ', num2str(mean(balanced_accuracy), '%.15f')]);

disp(['Marco F-Beta Score: ', num2str(mean(marco_f_beta_score) , '%.15f')]);
disp(['Mirco F-Beta Score: ', num2str(mean(mirco_f_beta_score) , '%.15f')]);
disp(['Weighted F-Beta Score: ', num2str(mean(weighted_f_beta_score) , '%.15f')]);

disp(['Macro Jaccard: ', num2str(macro_jaccard, '%.15f')]);
disp(['Micro Jaccard: ', num2str(maicro_jaccard, '%.15f')]);
disp(['Weighted Jaccard: ', num2str(weighted_Jaccard, '%.15f')]);
