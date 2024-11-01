clc;
clear;
close all;
%% Data Loading:
data = readtable('Corr_output.xlsx', 'Sheet','morph_av-ngl_glnu');
x = data.x;
y = data.y;
%binary_array = data.binary;

%% Calculating Evaluation Metrics:

pearson = corr(x, y, 'type','Pearson');
disp(['Pearson: ', num2str(pearson, '%.15f')]);

spearman = corr(x, y, 'Type', 'Spearman');
disp(['Spearman:', num2str(spearman, '%.15f')]);

kendall = corr(x, y, 'Type', 'Kendall');
disp(['Kendall: ', num2str(kendall, '%.15f')]);

% point_biserial = corr(binary_array, y);
% disp(['Point Biserial:', num2str(point_biserial, '%.15f')]);

dcor = dist_corr(x, y);
disp(['Distance:', num2str(dcor, '%.15f')]);


mutual_information = mutualInformationDiscrete(x, y);
disp(['Mutual Information:', num2str(mutual_information, '%.15f')]);

r = bicor(x, y);
disp(['Bicorrelation coefficient: ', num2str(r, '%.15f')]);



%% Functions:

function bicor_corr = bicor(x, y)
    x = x(:);
    y = y(:);

    rankX = tiedrank(x);
    rankY = tiedrank(y);

    bicor_corr = corr(rankX, rankY);
end




function mi = mutualInformationDiscrete(X, Y)
    % Ensure X and Y are column vectors
    X = X(:);
    Y = Y(:);
    
    % Create joint frequency table
    jointTable = crosstab(X, Y);
    jointProb = jointTable / sum(jointTable(:));
    
    % Marginal probabilities
    pX = sum(jointProb, 2);
    pY = sum(jointProb, 1);
    
    % Avoid division by zero and log of zero
    jointProb(jointProb == 0) = eps;
    pX(pX == 0) = eps;
    pY(pY == 0) = eps;
    
    % Mutual information calculation
    mi = sum(sum(jointProb .* log(jointProb ./ (pX * pY))));
end








