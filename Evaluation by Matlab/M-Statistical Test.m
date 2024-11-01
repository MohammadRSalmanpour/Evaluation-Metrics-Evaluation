clc;
clear;
close all
%% Data Loading:
data = readtable('stest.xlsx', 'Sheet','Sheet1');
x = data.x;
y = data.y;

%% Independent t-test:

[~, p_value1, ~, stats1] = ttest2(x, y);
fprintf('<strong>Independent t-test::</strong>\n')
disp(['itt-statistic: ', num2str(stats1.tstat, '%.15f')]);
disp(['itt_p-value: ', num2str(p_value1, '%.15f')]);

%% Paired t-test:

[~, p_value2, ~, stats2] = ttest(x, y);
fprintf('<strong>Paired t-test:</strong>\n');
disp(['ptt-statistic: ', num2str(stats2.tstat, '%.15f')]);
disp(['ptt_p-value: ', num2str(p_value2, '%.15f')]);

%% Kolmogorov-Smirnov Test:

[~, p_value3, stats3] = kstest2(x, y);
fprintf('<strong>Kolmogorov-Smirnov Test:</strong>\n');
disp(['kst-statistic: ', num2str(stats3, '%.15f')]);
disp(['kst_p-value: ', num2str(p_value3, '%.15f')]);

%% Chi2:
observed = [30, 10, 40, 20];
expected = [25, 25, 25, 25];
[~, p_value4, stats4,] = chi2gof(expected, 'Expected', observed);
fprintf('<strong>Chi2:</strong>\n');
disp(['Chi-squared statistic: ', num2str(stats4.chi2stat, '%.15f')]);
disp(['Chi-squared_p-value: ', num2str(p_value4, '%.15f')]);

%% ANOVA:

data = [x; y];
group = [repmat({'Group1'}, length(x), 1); 
         repmat({'Group2'}, length(y), 1)];
[p_value5, tbl, stats5] = anova1(data, group);

fprintf('<strong>ANOVA:</strong>\n');
disp(['ANOVA_p_value: ', num2str(p_value5, '%.15f')]);
disp(stats5);

%% Kruskal-Wallis Test / Mann-Whitney U Test:

[p_value6, ~, stats6] = ranksum(y, x);
fprintf('<strong>Mann-Whitney U Test Results:</strong>\n');
disp(['mwut_statistic: ', num2str(stats6.ranksum, '%.15f')]);
disp(['mwut_p-value: ', num2str(p_value6, '%.15f')]);

%% Shapiro-Wilk Test:
[H, pValue_se, W] = swtest(x);
fprintf('<strong>Shapiro-Wilk Test:</strong>\n');
disp(['swt_statistic: ', num2str(W, '%.15f')]);
disp(['swt_p-value: ', num2str(pValue_se, '%.15f')]);

%% F-test:

[~, p_value7, ~, stats7] = vartest2(x, y);
stats7 = tbl{2,5};

fprintf('<strong>F-Test:</strong>\n');
disp(['F-statistic: ', num2str(stats7, '%.15f')]);
disp(['F-p-value: ', num2str(p_value7, '%.15f')]);

%% PT

mu = 0;
[~, p, ~, stats] = ttest(data, mu);

t_statistic = stats.tstat;

fprintf('<strong>One-Sample t-Test Results:</strong>\n');
disp(['t-statistic: ', num2str(t_statistic, '%.15f')]);
disp(['p-value: ', num2str(p, '%.15f')]);


%% Welch's t-test:

[~, p_value8, ci, stats8] = ttest2(x, y, 'Vartype', 'unequal');

fprintf('<strong>Walchs t-test:</strong>\n');
disp(['wet-statistic: ', num2str(stats8.tstat, '%.15f')]);
disp(['wet-p-value: ', num2str(p_value8, '%.15f')]);

%% Bartlett's test:

[p_value10, stats10] = vartestn(data, group, 'TestType', 'Bartlett');
fprintf('<strong>Bartletts Test:</strong>\n');
disp(['bt-statistic: ', num2str(stats10.chisqstat, '%.15f')]);
disp(['bt-p-value: ', num2str(p_value10, '%.15f')]);

%% Levene's test:

[p_value9, stats9] =  vartestn(data, group, 'TestType', 'LeveneAbsolute', 'Display', 'off');
fprintf('<strong>Levenes Test:</strong>\n');
disp(['lt-statistic: ', num2str(stats9.fstat, '%.15f')]);
disp(['lt-p-value: ', num2str(p_value9, '%.15f')]);



%% Wilcoxon Signed-Rank Test:

[p_value11, h, stats11] = signrank(x, y);
fprintf('<strong>Wilcoxon Signed-Rank Test:</strong>\n');
disp(['wstr-statistic: ', num2str(stats11.signedrank , '%.15f')]);
disp(['wstr-p-value: ', num2str(p_value11, '%.15f')]);

%% Likelihood Ratio:

full_model = fitlm(x, y);
reduced_model = fitlm(x(:,1), y);

ll_full = logLikelihood(full_model, y, x);
ll_reduced = logLikelihood(reduced_model, y, x(:,1));

lrt_statistic = -2 * (ll_reduced - ll_full);

df = full_model.NumCoefficients - reduced_model.NumCoefficients; 
p_value = 1 - chi2cdf(lrt_statistic, df);

fprintf('<strong>Likelihood Ratio Test Results:</strong>\n');
disp(['LRT Statistic: ', num2str(lrt_statistic, '%.15f')]);
disp(['LRT_p-value: ', num2str(p_value, '%.15f')]);


%% Z Test:

[z_stat, z_p_value] = twoSampleZTest(x, y);
fprintf('<strong>Z Test:</strong>\n');
disp(['z-statistic: ', num2str(z_stat, '%.15f')]);
disp(['z-p-value: ', num2str(z_p_value, '%.15f')]);

%% Mean Difference:

mean_difference = mean(x)- mean(y);
fprintf('<strong>Mean Difference:</strong>\n');
disp(['mean_diff: ', num2str(mean_difference, '%.15f')]);

%% Standard Deviation Difference:

std_diff = std(x) - std(y);
fprintf('<strong>Standard Deviation Difference:</strong>\n');
disp(['std_diff: ', num2str(std_diff, '%.15f')]);


%% Correlation:

correlation = corr(x, y);
fprintf('<strong>Correlation:</strong>\n');
disp(['corr: ', num2str(correlation, '%.15f')]);



%% Functions:

function [zStat, pValue] = twoSampleZTest(sample1, sample2)
    % Two-Sample Z-Test
    % Input:
    %   - sample1: Vector of data for the first sample
    %   - sample2: Vector of data for the second sample
    % Output:
    %   - zStat: Z-statistic
    %   - pValue: p-value of the test

    % Calculate sample means and sizes
    mean1 = mean(sample1);
    mean2 = mean(sample2);
    n1 = length(sample1);
    n2 = length(sample2);

    var1 = var(sample1);
    var2 = var(sample2);
    pooledStdDev = sqrt((var1 / n1) + (var2 / n2));
    zStat = (mean1 - mean2) / pooledStdDev;
    pValue = 2 * (1 - normcdf(abs(zStat)));

end




function ll = logLikelihood(model, y, X)
    yhat = predict(model, X);
    residuals = y - yhat;
    ll = -0.5 * (length(y) * log(2*pi) + sum(residuals.^2) / model.MSE);
end


function p_value = permutation_test(x, y, n_permutations)
    % Inputs:
    % x: data vector 1
    % y: data vector 2
    % n_permutations: number of random permutations
    combined_data = [x; y];
    obs_stat = mean(x) - mean(y);
    perm_stats = zeros(n_permutations, 1);

    for i = 1:n_permutations
        shuffled_data = combined_data(randperm(length(combined_data)));
        perm_x = shuffled_data(1:length(x));
        perm_y = shuffled_data(length(x)+1:end);
        perm_stats(i) = mean(perm_x) - mean(perm_y);
    end
    p_value = mean(abs(perm_stats) >= abs(obs_stat));
    fprintf('P-value: %.4f\n', p_value);
end

