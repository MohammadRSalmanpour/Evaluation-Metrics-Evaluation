clc;
clear;
close all

%% Data 1:
% y_true = [3.0; 0.5; 2.0; 7.0; 2.4; 3.7; 6.2];
% y_pred = [2.5; 0.0; 2.0; 8.0; 2.5; 2.0; 6.0];

% Data 2:
data = readtable('regression_data.xlsx', 'Sheet', 'HC_DF_SPT_Head and Neck');
y_true = data.y_true;
y_pred = data.y_pred;

%% Calculating Evaluation Metrics:

MAE = mae(y_true, y_pred);
disp(['Mean Absolute Error: ', num2str(MAE, '%.15f')]);
disp(['Mean Absolute Error STD: ', num2str(std(y_true - y_pred), '%.15f')]);


MSE = immse(y_true, y_pred);
disp(['Mean Squared Error: ', num2str(MSE, '%.15f')]);
disp(['Mean Squared Error STD: ', num2str(std((y_true - y_pred).^2), '%.15f')]);


RMSE = rmse (y_true, y_pred);
disp(['Root Mean Squared Error: ', num2str(RMSE, '%.15f')]);



r_squared = 1- (sum((y_true - y_pred).^2) / sum((y_true - mean(y_pred)).^2));
% lm = fitlm(y_true, y_pred);
% r_squared = lm.Rsquared.Ordinary;
disp(['R Squared: ', num2str(r_squared, '%.15f')]);

MAPE = mean(abs((y_true - y_pred) ./ y_true) * 100);
disp(['Mean Absolute Percentage Error: ', num2str(MAPE, '%.15f')]);
disp(['Mean Absolute Percentage Error STD: ', num2str(std((y_true - y_pred) ./ y_true), '%.15f')]);

MSLE = mean(  (log(1 + y_true) - log(1  + y_pred)).^2 );
disp(['Mean Squared Logarithmic Error: ', num2str(MSLE, '%.15f')]);
disp(['Mean Squared Logarithmic Error STD: ', num2str(std((log(1 + y_true) - log(1  + y_pred)).^2) , '%.15f')]);

explained_variance = 1- (var(y_true - y_pred) / var(y_true));
disp(['Explained Variance: ', num2str(explained_variance,'%.25f')]);


medae = median(abs(y_true - y_pred));
disp(['Median Absolute Error: ', num2str(medae, '%.15f')]);


p = 0; 
deviance_value = TweedieDeviance(y_true, y_pred, p);
disp(['Tweedie deviance: ', num2str(mean(deviance_value), '%.15f')]);

delta = 1;
[loss, huber_loss] = HuberLoss(y_pred, y_true, delta);
disp(['Huber: ', num2str(loss, '%.15f')]);
disp(['Huber STD: ', num2str(std(huber_loss)/max(huber_loss), '%.15f')]);


%% Functions:

function tweedie_deviance = TweedieDeviance(y_true, y_pred, p)
    % Tweedie Deviance calculation, similar to scikit-learn's mean_tweedie_deviance
    % Inputs:
    %   y_true - vector of true target values
    %   y_pred - vector of predicted values
    %   p      - power parameter of Tweedie distribution (p=1.5 for typical Tweedie)

    assert(all(y_pred > 0), 'Predicted values must be positive for Tweedie deviance.');

    if p == 0
        tweedie_deviance = (y_true - y_pred).^2;
    elseif p == 1
        tweedie_deviance = 2 * (y_true .* log(y_true ./ y_pred) - (y_true - y_pred));
    elseif p == 2
        tweedie_deviance = 2 * (y_true - y_pred) ./ y_pred - log(y_true ./ y_pred);
    else
        tweedie_deviance = 2 * (y_true.^(2-p) - y_pred.^(2-p)) / ((1-p)*(2-p));
    end
    tweedie_deviance = mean(tweedie_deviance);
end


function [mean_huber_loss, huber_loss] = HuberLoss(y_true, y_pred, delta)
    y_true = y_true(:);
    y_pred = y_pred(:);
    residuals = abs(y_true - y_pred);
    huber_loss = zeros(size(y_true));
    huber_loss(residuals <= delta) = 0.5 * (residuals(residuals <= delta).^2);
    huber_loss(residuals > delta) = delta * (residuals(residuals > delta) - 0.5 * delta);
    mean_huber_loss = mean(huber_loss);
   
end


