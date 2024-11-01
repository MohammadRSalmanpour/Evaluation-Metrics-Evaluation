clc;
clear;
close all
%% Data Loading: 
data = readtable('I2I.xlsx', 'Sheet','Sheet1');
y_true = reshape(data.y_true, [128, 128]);
y_pred = reshape(data.y_pred, [128, 128]);
y_true_1D = y_true(:);
y_pred_1D = y_pred(:);

%% Evaluation Metrics:

MAE = mae(y_true, y_pred);
disp(['MAE: ', num2str(MAE, '%.15f')]);
disp(['Mean Absolute Error STD: ', num2str(std(y_true_1D - y_pred_1D), '%.15f')]);


MSE = immse(y_true, y_pred);
disp(['MSE: ', num2str(MSE, '%.15f')]);
disp(['Mean Squared Error STD: ', num2str(std((y_true_1D - y_pred_1D).^2), '%.15f')]);

RMSE = rmse (y_true_1D, y_pred_1D);
disp(['RMSE: ', num2str(mean(RMSE), '%.15f')]);

lm = fitlm(y_true_1D, y_pred_1D);
r_squared = lm.Rsquared.Ordinary;
disp(['R^2: ', num2str(r_squared, '%.15f')]);

[psnr, SNR] = psnr(y_pred, y_true);
disp(['PSNR: ', num2str(psnr, '%.15f'), ' dB']);

[ssimval, ssimmap] = ssim(y_pred, y_true);
disp(['SSIM:', num2str(ssimval, '%.15f')]);





















