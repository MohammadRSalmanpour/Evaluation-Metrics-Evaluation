clc;
clear;
close all
% Load data from the .mat file
data = load('random_3d_data.mat');
y_true = data.y_true;
y_pred = data.y_pred;

% Ensure y_true and y_pred are of size (30, 30, 30)
assert(all(size(y_true) == [30, 30, 30]), 'y_true must be of size (30, 30, 30)');
assert(all(size(y_pred) == [30, 30, 30]), 'y_pred must be of size (30, 30, 30)');

% Convert 3D arrays to double for calculations
y_true = double(y_true);
y_pred = double(y_pred);

%% Evaluation Metrics:

% Mean Absolute Error (MAE)
MAE = mean(abs(y_true(:) - y_pred(:)));
disp(['MAE: ', num2str(MAE, '%.15f')]);

% Standard deviation of the absolute error
disp(['Mean Absolute Error STD: ', num2str(std(abs(y_true(:) - y_pred(:))), '%.15f')]);

% Mean Squared Error (MSE)
MSE = immse(y_pred, y_true); % immse can work on multidimensional data directly
disp(['MSE: ', num2str(MSE, '%.15f')]);

% Standard deviation of the squared error
disp(['Mean Squared Error STD: ', num2str(std((y_true(:) - y_pred(:)).^2), '%.15f')]);

% Root Mean Squared Error (RMSE)
RMSE = sqrt(MSE);  % RMSE can be calculated directly from the MSE
disp(['RMSE: ', num2str(RMSE, '%.15f')]);

% R-squared (R^2) using linear model fit
lm = fitlm(y_true(:), y_pred(:));
r_squared = lm.Rsquared.Ordinary;
disp(['R^2: ', num2str(r_squared, '%.15f')]);

% Peak Signal-to-Noise Ratio (PSNR)
[psnr_value, SNR] = psnr(y_pred, y_true); % PSNR expects 3D arrays
disp(['PSNR: ', num2str(psnr_value, '%.15f'), ' dB']);

% Structural Similarity Index (SSIM)
[ssimval, ~] = ssim(y_pred, y_true); % SSIM for 3D images
disp(['SSIM: ', num2str(ssimval, '%.15f')]);
