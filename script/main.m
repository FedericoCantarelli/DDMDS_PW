%% DDSM project work earthquake
% The aim of this project is to find a model to describe seismic data of
% Kobe Earthquake in 1995 and to investigate whether it is possible, 
% from the model built on normal activity data, to predict the earthquake 
% measurement.

% All cells are written with a pause at the end so to go through the script
% just press enter
clear all
close all
clc


%% Load Data
% Load data from mat file and plot them
data = load("earthquake.mat");

plot(1:3048, data.kobe)
title("Earthquake Activity", "from Kobe, 1995")
xlim([1 3048])

pause
close all


%% Split data in normal, transition and earthquake activity
% Steps of different seismic stages given given a priori

normal_activity = data.kobe(1:1200);
transition_activity = data.kobe(1201:1700);
earthquake_activity = data.kobe(1701:end);

% Plot the data
figure
plot(1:1200, normal_activity, 1201:1700, transition_activity, 1701:3048, earthquake_activity);
legend("Normal Activity", "Transition Activity", "Real Activity", Location="northwest");
title("Earthquake Activity", "from Kobe, 1995");
xlim([1 3048]);

pause
close all


%% Boxplot for the different activities

% Create a dummy variable to group activity
g1 = repmat({'Normal Activity'},1200,1); 
g2 = repmat({'Transition Activity'},500,1);
g3 = repmat({'Real Activity'},1348,1);
g = [g1; g2; g3];

figure
boxplot(data.kobe, g, Colors="k", MedianStyle="line", Symbol="or");
title("Earthquake Activity", "from Kobe, 1995");

pause
close all

%% Check the variance
% I want to check if variances in the different phases of the earthquake
% are different. To do so, if all the sample are from a normal
% distribution I will use the Bartlett's test, otherwise I  will use
% Levene's test, which has a more relaxed assumption on normality of the 
% data. To check samples normality I will perform a Anderson-Darling test 
% for normality. Let's use Bonferroni to check if all data are from a 
% normal distribution

sig_lev = 0.05/3;

disp("Normal activity adtest");
[h,p] = adtest(normal_activity,"Alpha",sig_lev)
pause

disp("Transition activity adtest");
[h,p] = adtest(transition_activity, "Alpha",sig_lev)
pause

disp("Real activity adtest");
[h,p] = adtest(earthquake_activity, "Alpha",sig_lev)

pause
clc

% The last sample has a really low p-value so we can reject the null
% hypothesis, hence the data are not normal. We are going to
% use Levene's Test


%% Levene's test
[p,stats] = vartestn(data.kobe, g, 'TestType','LeveneAbsolute');
pause

close all
clc

%% Summary statistics of normal activity
M = mean(normal_activity);
disp("Mean for normal activity:");
disp(M);
pause

V = var(normal_activity);
disp("Variance for normal activity:");
disp(V);
pause

% From the GUI of the plot I fitted a linear trend line in order to check
% if the trend is constant or not
% Tools > Basic fitting > Linear and check equation and R^2 boxes
figure
plot(1:1200, normal_activity)
title("Normal Activity", "from Kobe, 1995")
pause
close all

% From the linear trend we can see that the mean is constant

unbiased_normal = normal_activity-M;

figure
plot(1:1200, normal_activity, 1:1200, unbiased_normal);
legend("Normal Activity", "Unbiased Normal Activity", Location="northwest");
title("Earthquake Activity", "from Kobe, 1995");
xlim([1 1200]);
pause

close all
clc


%% Periodogram, covariance function, ACF and PACF plots of timeserie
[Pxx, w] = periodogram(normal_activity);
[Pyy, z] = pwelch(normal_activity,bartlett(100),0);


% Plot the spectrum of the output
figure
plot(w/pi, 10*log10(Pxx), "Color","#000000");
hold on
plot(z/pi, 10*log10(Pyy), LineWidth= 2, Color="red");
hold on
title("Spectrum for Normal Activity", "Earthquake from Kobe, 1995");
legend("Periodogram for Normal Activity", "Smoothed Periodogram for Normal Activity", Location="southwest");
pause
close all

% From the spectrum we can see that the signal is a low-frequency pass
% filter


% Plot the covariance function
gamma = covf(unbiased_normal, 50);
figure
plot(0:49,gamma, Marker=".", MarkerSize=20, LineWidth=1.2);
title("Covariance Function", "Earthquake from Kobe, 1995");
pause
close all

% ACF and PACF
[xc,lags] = autocorr(normal_activity);
[pxc,plags] = parcorr(normal_activity);

L = length(normal_activity);
vcrit = sqrt(2)*erfinv(0.95);
lconf = -vcrit/sqrt(L);
upconf = vcrit/sqrt(L);

figure
% Top plot
nexttile
stem(lags,xc,'filled', LineWidth=0.75);
hold on
plot(lags, xc, LineStyle="none", Color = "#0072BD", Marker=".", MarkerSize=20);
hold on
plot(lags,lconf*ones(size(lags)),'--r');
hold on
plot(lags,upconf*ones(size(lags)),'--r');
title('Autocorrelation Function with 95% Confidence Intervals', FontSize=15, FontWeight='normal');
hold off

% Bottom plot
nexttile
stem(plags,pxc,'filled', LineWidth=0.75);
hold on
plot(plags,pxc, LineStyle="none", Color = "#0072BD", Marker=".", MarkerSize=20);
hold on
plot(plags,lconf*ones(size(lags)),'--r');
hold on
plot(plags,upconf*ones(size(lags)),'--r');
title('Autocorrelation Function with 95% Confidence Intervals', FontSize=15, FontWeight='normal');
hold off

pause
close all

% From the ACF plot (which is just a standardized version of the covariance
% function) and the PACF plot, I can conclude that AR and MA model are not
% appropriate to describe the process, therefore I need an ARMA model. 
% The reason why is that ACF and PACF aren't null after a large number of 
% lags.


%% Train-Validation Split
train_perc = 0.8;
normal_activity_train=unbiased_normal(1:1200*0.8);
normal_activity_test=unbiased_normal(1200*0.8+1:end);

x1 = 1:1200*train_perc;
x2 = 1200*train_perc+1:1200;

figure
plot(x1, normal_activity_train, x2, normal_activity_test);
legend("Train", "Test", Location="southwest");
title("Normal Activity", "from Kobe, 1995");
xlim([1 1200]);

pause
close all

%% Load data in iddata structure for model training
my_data_train = iddata(normal_activity_train, Ts = 1);
my_data_test = iddata(normal_activity_test, Ts = 1);

%% Model training
% To avoid time consuming computations, i trained models once
% If needed you can load the enviroment model_fitted_40.mat directly from
% the next cell
na = 1:40;
nc = 1:40;
ct = 1;
models = cell(1,length(na)*length(nc));
for i = 1:length(na)
    na_ = na(i);
    for j = 1:length(nc)
        nc_ = nc(j); 
        models{ct} = armax(my_data_train,[na_ nc_]);
        ct = ct+1;
    end
end

%% Save trained model / load backup
% If you want to retrain models from scratch, just run the cell above and
% change the command load into save
load("model_fitted_40.mat");

%% Select best model
% As before, if needed you can load the environment
% best_model_found_validation.mat instead of running the entire cell

% Make Prediction
y_hat = predict(models{1}, my_data_train, 1);

% Init MSE
min_MSE = computeMSE(my_data_train.y, y_hat.y);

% Init RMSE
min_RMSE = sqrt(min_MSE);

% Init MAE
min_MAE = computeMAE(my_data_train.y, y_hat.y);

% Init AIC
min_AIC = computeAIC(my_data_train.y, y_hat.y, length(models{1}.A), length(models{1}.C), false);

% Init AIC
min_AIC_corr = computeAIC(my_data_train.y, y_hat.y, length(models{1}.A), length(models{1}.C), true);


% Init MDL
min_MDL = computeMDL(my_data_train.y, y_hat.y, length(models{1}.A), length(models{1}.C));

% Init FPE
min_FPE = computeFPE(my_data_train.y, y_hat.y, length(models{1}.A), length(models{1}.C));

% Init MAPE
min_MAPE = computeMAPE(my_data_train.y, y_hat.y);

% Init Fit Percentage
max_FP = models{1}.Report.Fit.FitPercent;

for i = 1:(length(na)*length(nc))
    y_hat = predict(models{i}, my_data_train, 1);

    current_MSE = computeMSE(my_data_train.y, y_hat.y);
    if current_MSE < min_MSE
        min_MSE = current_MSE;
        index_min_MSE = i;
    end


    current_RMSE = sqrt(current_MSE);
    if current_RMSE < min_RMSE
        min_RMSE = current_RMSE;
        index_min_RMSE = i;
    end


    current_MAE = computeMAE(my_data_train.y, y_hat.y);
    if current_MAE < min_MAE
        min_MAE = current_MAE;
        index_min_MAE = i;
    end


    current_AIC = computeAIC(my_data_train.y, y_hat.y, length(models{i}.A), length(models{i}.C), false);
    if current_AIC < min_AIC
        min_AIC = current_AIC;
        index_min_AIC = i;
    end


    current_AIC_corr = computeAIC(my_data_train.y, y_hat.y, length(models{i}.A), length(models{i}.C), true);
    if current_AIC_corr < min_AIC_corr
        min_AIC_corr = current_AIC_corr;
        index_min_AIC_corr = i;
    end


    current_MDL = computeMDL(my_data_train.y, y_hat.y, length(models{i}.A), length(models{i}.C));
    if current_MDL < min_MDL
        min_MDL = current_MDL;
        index_min_MDL = i;
    end

    current_FPE = computeFPE(my_data_train.y, y_hat.y, length(models{i}.A), length(models{i}.C));
    if current_FPE < min_FPE
        min_FPE = current_FPE;
        index_min_FPE = i;
    end


    current_MAPE = computeMAPE(my_data_train.y, y_hat.y);
    if current_MAPE < min_MAPE
        min_MAPE = current_MAPE;
        index_min_MAPE = i;
    end


    current_FP = models{i}.Report.Fit.FitPercent;
    if current_FP > max_FP
        max_FP = current_FP;
        index_max_FP = i;
    end

end

%% Unpack Model
model_min_MSE = models{index_min_MSE};
model_min_RMSE = models{index_min_RMSE};
model_min_AIC = models{index_min_AIC};
model_min_AIC_corr = models{index_min_AIC_corr}; 
model_min_MDL = models{index_min_MDL};
model_min_MAE = models{index_min_MAE};
model_min_FPE = models{index_min_FPE};
model_min_MAPE = models{index_min_MAPE};
model_max_FP = models{index_max_FP};

%% Save for backup / load backup
% If you want to rerun the procedure to find the best models from scratch, 
% just run the cell above and change the command load into save
load("best_model_found.mat");

%% Check model selected with adjusted AIC
compare(my_data_train, model_min_AIC_corr,1);
pause
close all

resid(my_data_train, model_min_AIC_corr);
pause
close all

% The percentage of fitting is very good, but residual are still slightly
% corretalted though.

%% Model Residuals
% My initial idea was to choose the best model in terms of trade-off
% between model complexity and model performances and the above mentioned
% model was the simplest one. Unfortunately, the model is so simple that is
% not able to capture all the dynamics of the process
% At the end, I chose the model found with the MDL criterion. Indeed, the
% residuals of the model were slightly correlated at lag 22, but we know
% that for higher lags the rule is relaxed. Furthermore, we have very few 
% data so the rules is, once again, relaxed.
% I decided to keep using the minum MDL model, so the optimal model is an 
% ARMA(9,4). By making thgis choice, I tried also to avoid overfitting on 
% new data. In this section I will check model residuals and in the next 
% one, I will check if the model is overfitting the data.

optimal_model = model_min_MDL;

compare(my_data_train, optimal_model, 1);
pause
close all

e = resid(my_data_train, optimal_model);
resid(my_data_train, optimal_model);
pause
close all


[Pxx2, w2] = periodogram(e.y);
[Pyy2, z2] = pwelch(e.y,bartlett(100),0);


% Plot the spectrum of the residuals
figure
plot(w2/pi, 10*log10(Pxx2), "Color","#000000");
hold on
plot(z2/pi, 10*log10(Pyy2), LineWidth= 2, Color="red");
hold on
title("Spectrum for Residuals", "MDL criterion");
legend("Periodogram for Residuals", "Smoothed Periodogram for Residuals");
pause
close all


% The function whiteness_test is shared by Carlos Mario Vélez S. on
% mathworks file exchange and it plots a minitab 4-in-1 style plot for
% residuals

e = resid(my_data_train, model_min_MDL);
whiteness_test(e.y)
pause
close all

% t-test on residuals
disp("T-test on residuals");
[h,p,ci] = ttest(e.y)
pause
clc

% qqplot 
p = qqplot(e.y);
set(p(1),'marker','.','markersize',14,'markeredgecolor',[0 0.4470 0.7410]);
pause
close all

% We can see that the tails exibhit a strange behaviour, but using the
% inspection plot tool we can see that those critical points are related to
% outliers in the original signal. So, I decided to perform two AD test for
% normality, one with all the residuals and one removing theese critical
% points.

% Plot of normal activity
figure
plot(normal_activity_train, "-k");
hold on 
yline(mean(normal_activity_train)+2*sqrt(var(normal_activity_train)), "LineStyle","-.", "LineWidth", .9, "Color", "blue");
hold on 
yline(mean(normal_activity_train)+3*sqrt(var(normal_activity_train)), "LineStyle","--", "LineWidth", .9, "Color", "red");
hold on 
yline(mean(normal_activity_train)-2*sqrt(var(normal_activity_train)), "LineStyle","-.", "LineWidth", .9, "Color", "blue");
hold on 
yline(mean(normal_activity_train)-3*sqrt(var(normal_activity_train)), "LineStyle","--", "LineWidth", .9, "Color", "red");
hold on
xlim([1 960]);
title("Earthquake Activity", "from Kobe, 1995")
pause
close all

% ad-test residuals
disp("adtest on residuals:")
[h, p] = adtest(e.y)
pause
clc

% ad-test residuals without critical points
disp("adtest on residuals without critical points:")
temp_err = e.y;
temp_err([124 121 152 948 130 145 802 145 176 125 745 384 226]) = [];
[h, p] = adtest(temp_err)
pause
clc


% ECDF of residuals
mu = mean(e.y);
sigma = sqrt(var(e.y));
x = linspace(min(e.y), max(e.y), length(e.y));
y = cdf('Normal',x,mu,sigma);

[f,step] = ecdf(e.y);

figure
plot(step, f, LineWidth=1.5);
hold on
plot(x, y, LineWidth=1.5);
hold on 
legend("Estimated CDF", "Teoretical CDF", Location="bestoutside");
hold on 
title("Residuals", "Cumulative distribution function", FontSize=14);
hold on
xlim([-inf inf]);
pause
close all


%% Model Validation
compare(optimal_model, my_data_test,1);
pause
close all

disp("Model:Training data");
y_hat = predict(optimal_model, my_data_train, 1);
disp("Mean Squared Error");
computeMSE(my_data_train.y, y_hat.y)
pause

disp("Root Mean Squared Error");
sqrt(computeMSE(my_data_train.y, y_hat.y))
pause

disp("AIC");
computeAIC(my_data_train.y, y_hat.y, length(optimal_model.A), length(optimal_model.C), false)
pause

disp("Mean Absolute Error");
computeMAE(my_data_train.y, y_hat.y)
pause

disp("Mean Absolute Percentage Error");
computeMAPE(my_data_train.y, y_hat.y)
pause

disp("Minimum Description Length");
computeMDL(my_data_train.y, y_hat.y, length(optimal_model.A), length(optimal_model.C))
pause

disp("Model: Training data");
y_hat = predict(optimal_model, my_data_test, 1);
disp("Mean Squared Error");
computeMSE(my_data_test.y, y_hat.y)
pause

disp("Root Mean Squared Error");
sqrt(computeMSE(my_data_test.y, y_hat.y))
pause

disp("AIC");
computeAIC(my_data_test.y, y_hat.y, length(optimal_model.A), length(optimal_model.C), false)
pause

disp("Mean Absolute Error");
computeMAE(my_data_test.y, y_hat.y)
pause

disp("Mean Absolute Percentage Error");
computeMAPE(my_data_test.y, y_hat.y)
pause

disp("Minimum Description Length");
computeMDL(my_data_test.y, y_hat.y, length(optimal_model.A), length(optimal_model.C))
pause
clc

%% Validation residuals
optimal_model = model_min_MDL;

compare(my_data_test, optimal_model, 1);
pause
close all

% We can see that the residuals on validation set are slightly more
% correlated then the ones on training set. This could be due to the
% different variance of training set and validation set and also to small
% dimension of both our training set and validation set. The AD test for
% normality cannot reject H0 and the qqplot it's almost perfect (once again
% the tails are the outlier of the original signal)
e = resid(my_data_test, optimal_model);
resid(my_data_test, optimal_model);
pause
close all


[Pxx2, w2] = periodogram(e.y);
[Pyy2, z2] = pwelch(e.y,bartlett(100),0);


e = resid(my_data_test, optimal_model);
whiteness_test(e.y)
pause
close all

% t-test on residuals
disp("T-test on residuals");
[h,p,ci] = ttest(e.y)
pause
clc

% adtest on residuals
disp("Anderson-Darling test on residuals");
[h,p] = adtest(e.y)
pause
clc

% qqplot 
p = qqplot(e.y);
set(p(1),'marker','.','markersize',14,'markeredgecolor',[0 0.4470 0.7410]);
pause
close all

%% Compare model fit percentage
compare(my_data_test, optimal_model, 1);
pause
close all

%% Model Stability
num = optimal_model.C;
den = optimal_model.A;
sys = tf(num, den ,1);
pzplot(sys);

pause
close all
%% Compute me = 1/W(1)*my
% I want to express the model as black box model. Thus i need to find me
% starting from signal bias and the new estimated transfer function.
num = sum(cell2mat(sys.Numerator));
den = sum(cell2mat(sys.Denominator));
me = M*den/num;
disp("Error mu");
disp(me);
pause
clc


%% Model vs Prediction horizon (train and test data)
FPE_model = cell(50);
for i = 1:50
    y_hat = predict(my_data_train, optimal_model, i);
    FPE_model{i} = computeFPE(my_data_train.y, y_hat.y, length(optimal_model.A), length(optimal_model.C));
end

figure
plot(1:50, cell2mat(FPE_model), "-k", "LineWidth",1);
hold on
yline(var(my_data_train.y), "--r", "LineWidth",1);
hold on 
title("FPE of Training Data", "FontSize",14);
hold on 
xlabel("Prediction Horizon");
ylabel("FPE");
legend("Final Prediction Error", "Variance of training data", "Location","southeast");

pause
close all


for i = 1:50
    y_hat = predict(my_data_test, optimal_model, i);
    FPE_model{i} = computeFPE(my_data_test.y, y_hat.y, length(optimal_model.A), length(optimal_model.C));
end


figure
plot(1:50, cell2mat(FPE_model), "-k", "LineWidth",1);
hold on
yline(var(my_data_train.y), "--r", "LineWidth",1);
hold on
yline(var(my_data_test.y), "--b", "LineWidth",1);
hold on 
title("FPE of Validation Data", "FontSize",14);
hold on 
xlabel("Prediction Horizon");
ylabel("FPE");
legend("Final Prediction Error", "Variance of Training data", "Variance of Validation data", "Location","southeast");

pause
close all

%% Error to signal ratio
ets_model = cell(50);
for i = 1:50
    y_hat = predict(my_data_train, optimal_model, i);
    ets_model{i} = var(my_data_train.y-y_hat.y)/var(my_data_train.y);
end

ets_model_val = cell(50);
for i = 1:50
    y_hat = predict(my_data_test, optimal_model, i);
    ets_model_val{i} = var(my_data_test.y-y_hat.y)/var(my_data_test.y);
end

figure
plot(1:50, cell2mat(ets_model), Color='red', LineWidth=1, Marker='o', MarkerSize = 6, MarkerFaceColor='#FFFFFF');
hold on 
plot(1:50, cell2mat(ets_model_val), Color='blue', LineWidth=1, Marker='o', MarkerSize = 6, MarkerFaceColor='#FFFFFF');
hold on
legend("ESR train data", "ESR validation data");
hold on
title("Error to Signal Ratio", "Versus Prediction Horizon");
pause 
close all

%% Transition activity
% I want to investigate if we can somehow predict an earthquake. The idea
% is really simple: I built a model of normal activity dynamics so if the
% error increases, this means system dynamics changed. I decided to split
% data in time windows of 100 sec, because I think that system dynamics
% will change more and more over time, and they will be completely
% different during earthquake peak. Since the time series has an
% oscillatory component (due to how seismic data are collected) I decided
% to use MSE

% Create the unbiased version of transition data
unbiased_transition = transition_activity - M;
window_size = 100;

group_n = floor(length(unbiased_transition)/window_size);
mse_array = cell(group_n);



for i = 0:(group_n-1)
    t = iddata(unbiased_transition(i*window_size+1:(i+1)*window_size));
    y_ = predict(optimal_model, t, 1);
    mse_array{i+1} = computeMSE(t.y, y_.y);
end

y_hat_val = predict(optimal_model, my_data_test, 1);
mse_val = computeMSE(my_data_test.y, y_hat_val.y);
figure
plot(1:group_n, cell2mat(mse_array), '-o', 'MarkerSize', 10, "LineWidth", 1.2, "MarkerFaceColor", "#FFFFFF");
hold on 
yline(mse_val, '-r', "LineWidth", 1.2, "Color", "#D95319");
hold on 
legend("MSE Transition Data", "MSE Validation Data");
hold on 
title("Mean Squared Error",  "Transition Activity vs Validation Data", "FontSize",14);
pause
close all
clc


%% Earthquake activity
% Same as before but for earthquake activity
unbiased_eartquake = earthquake_activity - M;
window_size = 100;

group_n_eq = floor(length(unbiased_eartquake)/window_size);
mse_array_earthquake = cell(group_n_eq);


for i = 0:(group_n_eq-1)
    t = iddata(unbiased_eartquake(i*window_size+1:(i+1)*window_size));
    y_ = predict(optimal_model, t, 1);
    mse_array_earthquake{i+1} = computeMSE(t.y, y_.y);
end

y_hat_val = predict(optimal_model, my_data_test, 1);
mse_val = computeMSE(my_data_test.y, y_hat_val.y);
figure
plot(1:group_n_eq, cell2mat(mse_array_earthquake), '-o', 'MarkerSize', 10, "LineWidth", 1.2, "MarkerFaceColor", "#FFFFFF");
hold on 
yline(mse_val, '-r', "LineWidth", 1.2, "Color", "#D95319");
hold on 
legend("MSE Transition Data", "MSE Validation Data");
hold on 
title("Mean Squared Error",  "Earthquake vs Validation Data", "FontSize",14);
pause
close all
clc

%% Plot all together
y_hat_val = predict(optimal_model, my_data_test, 1);
mse_val = computeMSE(my_data_test.y, y_hat_val.y);
figure
plot(1:group_n, cell2mat(mse_array), 'k-o', 'MarkerSize', 10, "LineWidth", 1.2, "MarkerFaceColor", "#FFFFFF");
hold on
plot(group_n+1:group_n + group_n_eq, cell2mat(mse_array_earthquake), 'k-.d', 'MarkerSize', 10, "LineWidth", 1.2, "MarkerFaceColor", "#FFFFFF");
yline(mse_val, "LineWidth", 1.2, "Color", "red", "LineStyle","--");
hold on 
legend("MSE Transition Data", "MSE Earthquake Data", "MSE Validation Data");
hold on 
title("Mean Squared Error",  "Transition and Earthquake vs Validation Data", "FontSize",14);
hold on 
ylim([0 1.05*max(cell2mat(mse_array_earthquake))]);
hold on
xlim([0 19]);
hold off

pause
close all
clc

%% Third window of eartquake activity
% I would have expected a bell-like shape of MSE, and indeed it is
% There is just one let's say "outlier" in the third time window of the
% earthquake activity. Let's see if we can figure out the reason why.

g1 = 1:200;
g2 = 201:300;
g3 = 301:800;

window1 = earthquake_activity(g1);
window2 = earthquake_activity(g2);
window3 = earthquake_activity(g3);

figure
plot(g1, window1, "-k");
hold on
plot(g2, window2, "-r");
hold on
plot(g3, window3, "-k");
hold on
title("Third Window of EQ", "in red")
pause
close all

[y_hat,fit] = compare(optimal_model, iddata(window2),1);

figure
plot(window2, "-k");
hold on
plot(y_hat.y, "-r");
title("Third Window of EQ", strcat("Pct of fitting: ", num2str(round(fit,2)), "%"), "FontSize",16);
legend("Original data", "Predicted observations", Location="northwest")
pause 
close all


[y_hat,fit] = compare(optimal_model, iddata(window3),1);
figure
plot(window3, "-k");
hold on
plot(y_hat.y, "-r");
title("EQ from 300 to 800 sec", strcat("Pct of fitting: ", num2str(round(fit,2)), "%"), "FontSize",16);
legend("Original data", "Predicted observations", Location="northeast")
pause
close all

% It seems like in the interval 200-300 sec the earthquake activity has a
% dynamics more similar to transition and normal activity.


%% Conclusions
% From 400 seconds before eartquake beginning the MSE starts to increase,
% while from 200 seconds before eq beginning the MSE is sharply increased.
% So, we can predict with a 400-200 seconds precision the beginning of the
% eartquake.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Control chart approach
% The idea to build a real application system is the following: i decided
% to build a I-MR control chart on residuals to spot anomalies on the
% process.
% I have already checked residuals normality, and the test cannot slightly
% reject H0. Since I-MR charts are impacted by the normality, i decided to
% build the chart on sampled .

% Since there is one measurement per second, I decided to take just a
% random point every 10 points

   
% Compute residuals
optimal_model = model_min_MDL;
unbiased_transition = transition_activity - M;
unbiased_earthquake = earthquake_activity - M;

t_train_or = 1:960;
t_test_or = 961:1200;
t_trans_or = 1201:1700;
t_earth_or = 1701:3048;


y_hat = predict(my_data_train, optimal_model, 1).y;
e_train_or = my_data_train.y - y_hat;

y_hat = predict(my_data_test, optimal_model, 1).y;
e_test_or = my_data_test.y - y_hat;

y_hat = predict(iddata(unbiased_transition, TS=1), optimal_model, 1).y;
e_trans_or = unbiased_transition - y_hat;

y_hat = predict(iddata(unbiased_earthquake, Ts=1), optimal_model, 1).y;
e_earth_or = unbiased_earthquake - y_hat;


%% Random
WindowLength = 10;
t_train = zeros(floor(length(t_train_or)/WindowLength),1);
e_train = zeros(floor(length(e_train_or)/WindowLength),1);

for idx = 1:floor(length(t_train_or)/WindowLength)
    Block_e = e_train_or((idx-1)*WindowLength+1:idx*WindowLength);
    Block_t= t_train_or((idx-1)*WindowLength+1:idx*WindowLength);
    i = randi([1 WindowLength], 1, 1);
    t_train(idx) = Block_t(i);
    e_train(idx) = Block_e(i);
end


t_test = zeros(floor(length(t_test_or)/WindowLength),1);
e_test = zeros(floor(length(e_test_or)/WindowLength),1);

for idx = 1:floor(length(t_test_or)/WindowLength)
    Block_e = e_test_or((idx-1)*WindowLength+1:idx*WindowLength);
    Block_t= t_test_or((idx-1)*WindowLength+1:idx*WindowLength);
    i = randi([1 WindowLength], 1, 1);
    t_test(idx) = Block_t(i);
    e_test(idx) = Block_e(i);
end

t_trans = zeros(floor(length(t_trans_or)/WindowLength),1);
e_trans = zeros(floor(length(e_trans_or)/WindowLength),1);

for idx = 1:floor(length(t_trans_or)/WindowLength)
    Block_e = e_trans_or((idx-1)*WindowLength+1:idx*WindowLength);
    Block_t= t_trans_or((idx-1)*WindowLength+1:idx*WindowLength);
    i = randi([1 WindowLength], 1, 1);
    t_trans(idx) = Block_t(i);
    e_trans(idx) = Block_e(i);
end

t_earth = zeros(floor(length(t_earth_or)/WindowLength),1);
e_earth= zeros(floor(length(e_earth_or)/WindowLength),1);

for idx = 1:floor(length(t_earth_or)/WindowLength)
    Block_e = e_earth_or((idx-1)*WindowLength+1:idx*WindowLength);
    Block_t= t_earth_or((idx-1)*WindowLength+1:idx*WindowLength);
    i = randi([1 WindowLength], 1, 1);
    t_earth(idx) = Block_t(i);
    e_earth(idx) = Block_e(i);
end


%% Plot
figure
plot(t_train_or, e_train_or, "k-");
hold on
plot(t_test_or, e_test_or, "k-");
hold on
plot(t_trans_or, e_trans_or, "k-");
hold on
plot(t_earth_or, e_earth_or, "k-");
hold on
plot(t_train, e_train, LineStyle="none", Color="r", Marker=".", MarkerSize=10);
hold on
plot(t_test, e_test, LineStyle="none", Color="r", Marker=".", MarkerSize=10);
hold on
plot(t_trans, e_trans, LineStyle="none", Color="r", Marker=".", MarkerSize=10);
hold on
plot(t_earth, e_earth   , LineStyle="none", Color="r", Marker=".", MarkerSize=10);
title("Random Selected Points", FontSize=14);
hold on
xlim([1 3048]);
pause
close all


%% Compute mr
mr = zeros(length(e_train)-1,1);
for i=1:length(e_train)-1
    mr(i)=abs(e_train(i+1)-e_train(i));
end

%% I chart
alpha = 0.0027;
z = norminv(1-alpha/2);

I_CL = mean(e_train);
I_LCL = I_CL - z/1.128 * mean(mr);
I_UCL = I_CL + z/1.128 * mean(mr);


figure
plot(t_train, e_train, t_test, e_test, t_trans, e_trans, t_earth, e_earth);
hold on
yline(I_CL, "--g");
hold on
yline(I_LCL, "--r");
hold on
yline(I_UCL, "--r");
hold on
title("I-chart on batched residuals", FontSize=14);
xlim([1 3048]);
pause
close all

% Remove the 13th observation
e_train_no_13 = e_train;
e_train_no_13(13) = [];
t_train_no_13 = t_train;
t_train_no_13(13) = [];

mr = zeros(length(e_train_no_13)-1,1);
for i=1:length(e_train_no_13)-1
    mr(i)=abs(e_train_no_13(i+1)-e_train_no_13(i));
end

I_CL = mean(e_train_no_13);
I_LCL = I_CL - z/1.128 * mean(mr);
I_UCL = I_CL + z/1.128 * mean(mr);

out_of_control = (e_trans > I_UCL)|(e_trans < I_LCL);
ooc = e_trans(out_of_control);
tooc = t_trans(out_of_control);

figure
plot(t_train_no_13, e_train_no_13, LineStyle='-', Color="#377eb8");
hold on
plot(t_trans, e_trans, LineStyle='-', Color="#ff7f00");
hold on
plot(t_earth, e_earth, LineStyle='-', Color="#984ea3");
hold on
plot(tooc(1), ooc(1), LineStyle='none', Color="g", Marker="v", MarkerFaceColor="g", MarkerSize=8);
hold on
plot(tooc(2:end), ooc(2:end), LineStyle='none', Color="g", Marker="o", MarkerSize=5, LineWidth=1.5);
hold on
plot(t_test, e_test, LineStyle='-', Color="#377eb8");
hold on
yline(I_CL, LineStyle='-.', Color="#4daf4a", LineWidth=1);
hold on
yline(I_LCL, LineStyle='-.', Color="#e41a1c", LineWidth=1);
hold on
yline(I_UCL, LineStyle='-.', Color="#e41a1c", LineWidth=1);
hold on
title("I-chart on batched residuals", "10th observation excluded", FontSize=14);
hold on
legend("Normal activity", "Transition activity", "Earthquake activity", "First ooc", "Out of control");
hold on
xlim([1 3048]);
pause
close all

%% MR chart
MR_CL = mean(mr);
MR_LCL = 0;
MR_UCL = 3.267*MR_CL;

data = [e_train_no_13; e_test; e_trans; e_earth];
mr_2 = zeros(length(data)-1,1);
for i=1:length(data)-1
    mr_2(i)=abs(data(i+1)-data(i));
end

% Split for visualisation purposes
mr_train = mr_2(1:length(e_train_no_13));
mr_test = mr_2(1+length(e_train_no_13):length(e_train_no_13)+length(e_test));
mr_trans = mr_2(length(e_train_no_13)+length(e_test)+1:length(e_train_no_13)+length(e_test)+length(e_trans));
mr_earth = mr_2(length(e_train_no_13)+length(e_test)+length(e_trans)+1:length(e_train_no_13)+length(e_test)+length(e_trans)+length(e_earth)-1);

ooc = mr_trans(mr_trans > MR_UCL);
tooc = t_trans(mr_trans > MR_UCL);

figure
plot(t_train_no_13, mr_train, LineStyle='-', Color="#377eb8");
hold on
plot(t_trans, mr_trans, LineStyle='-', Color="#ff7f00");
hold on
plot(t_earth(1:end-1), mr_earth, LineStyle='-', Color="#984ea3");
hold on
plot(tooc(1), ooc(1), LineStyle='none', Color="g", Marker="v", MarkerFaceColor="g", MarkerSize=8);
hold on
plot(tooc(2:end), ooc(2:end), LineStyle='none', Color="g", Marker="o", LineWidth=1.5, MarkerSize=5);
hold on
plot(t_test, mr_test, LineStyle='-', Color="#377eb8");
hold on
yline(MR_CL, LineStyle='-.', Color="#4daf4a", LineWidth=1);
hold on
yline(MR_UCL, LineStyle='-.', Color="#e41a1c", LineWidth=1);
hold on 
yline(MR_LCL, LineStyle='-.', Color="#e41a1c", LineWidth=1);

hold on
legend("Normal activity", "Transition activity", "Earthquake activity", "First ooc", "Out of control");
hold on
xlim([1 3048]);
hold on
title("MR control chart", FontSize=14);
pause
close all

%% EWMA
lambda = 0.05;
L = 2.615;

mu = mean(e_train);
sigma = sqrt(var(e_train));
CL=mu;

data = [e_train; e_test; e_trans; e_earth];

EWMA = zeros(1+length(data),1);
EWMA(1) = mu;
UCL = zeros(length(data),1);
LCL = zeros(length(data),1);

for i=2:length(data)
    EWMA(i) = lambda*data(i) + (1-lambda)*EWMA(i-1);
    k = lambda/(2-lambda)*(1-(1-lambda)^(2*i));
    UCL(i) = CL + L*sigma*sqrt(k);
    LCL(i) = CL - L*sigma*sqrt(k);
end


CL = mu;
EWMA = EWMA(2:end);


% split for viz purpose
ewma_train = EWMA(1:length(e_train));
ewma_test = EWMA(length(e_train)+1:length(e_train)+length(e_test));
ewma_trans = EWMA(1+length(e_train)+length(e_test):length(e_train)+length(e_test)+length(e_trans));
ewma_earth = EWMA(length(e_train)+length(e_test)+length(e_trans)+1:length(e_train)+length(e_test)+length(e_trans)+length(e_earth));


% I can take the asymptotical UCL and LCL
out_of_control = (ewma_trans > UCL(end-1))|(ewma_trans < LCL(end-1));
ooc = ewma_trans(out_of_control);
tooc = t_trans(out_of_control);

figure
plot(t_train, ewma_train, LineStyle='-', Color="#377eb8");
hold on
plot(t_trans, ewma_trans, LineStyle='-', Color="#ff7f00");
hold on
plot(t_earth, ewma_earth, LineStyle='-', Color="#984ea3");
hold on
plot(tooc(1), ooc(1), LineStyle='none', Color="g", Marker="v", MarkerFaceColor="g", MarkerSize=8);
hold on
plot(tooc(2:end), ooc(2:end), LineStyle='none', Color="g", Marker="o", MarkerSize=5);
hold on
plot(t_test, ewma_test, LineStyle='-', Color="#377eb8");
hold on
yline(CL, LineStyle='-.', Color="#4daf4a", LineWidth=1);
hold on
plot([t_train; t_test; t_trans; t_earth], UCL, LineStyle='-.', Color="#e41a1c", LineWidth=1);
hold on 
plot([t_train; t_test; t_trans; t_earth], LCL, LineStyle='-.', Color="#e41a1c", LineWidth=1);
hold on
legend("Normal activity", "Transition activity", "Earthquake activity", "First ooc", "Out of control");
hold on
xlim([1 3048]);
hold on
ylim([-500 500]);
hold on
title("EWMA control chart", FontSize=14);
pause
close all


%% MA
mu = mean(e_train);
sigma = sqrt(var(e_train)); 
w = 5;

data = [e_train; e_test; e_trans; e_earth];
MA = zeros(length(data),1);
UCL = zeros(length(data),1);
LCL = zeros(length(data),1);

for i=1:w-1
    MA(i)=mean(data(1:i));
    UCL(i) = mu+3*sigma/sqrt(i);
    LCL(i) = mu-3*sigma/sqrt(i);
end


for i=w:length(data)
    MA(i)=mean(data(i-(w-1):i));
end

UCL(w:end)=mu+3*sigma/sqrt(5);
LCL(w:end)=mu-3*sigma/sqrt(5);

% Split for viz
ma_train = MA(1:length(e_train));
ma_test = MA(1+length(e_train):length(e_train)+length(e_test));
ma_trans = MA(length(e_train)+length(e_test)+1:length(e_train)+length(e_test)+length(e_trans));
ma_earth = MA(length(e_train)+length(e_test)+length(e_trans)+1:length(e_train)+length(e_test)+length(e_trans)+length(e_earth));


out_of_control = (ma_trans > UCL(end-1))|(ma_trans < LCL(end-1));
ooc = ma_trans(out_of_control);
tooc = t_trans(out_of_control);

figure
plot(t_train, ma_train, LineStyle='-', Color="#377eb8");
hold on
plot(t_trans, ma_trans, LineStyle='-', Color="#ff7f00");
hold on
plot(t_earth, ma_earth, LineStyle='-', Color="#984ea3");
hold on
plot(tooc(1), ooc(1), LineStyle='none', Color="g", Marker="v", MarkerFaceColor="g", MarkerSize=8);
hold on
plot(tooc(2:end), ooc(2:end), LineStyle='none', Color="g", Marker="o", LineWidth=1.5, MarkerSize=5);
hold on
plot(t_test, ma_test, LineStyle='-', Color="#377eb8");
hold on
yline(CL, LineStyle='-.', Color="#4daf4a", LineWidth=1);
hold on
plot([t_train; t_test; t_trans; t_earth], UCL, LineStyle='-.', Color="#e41a1c", LineWidth=1);
hold on 
plot([t_train; t_test; t_trans; t_earth], LCL, LineStyle='-.', Color="#e41a1c", LineWidth=1);
hold on
legend("Normal activity", "Transition activity", "Earthquake activity", "First ooc", "Out of control");
hold on
xlim([1 3048]);
hold on
title("MA control chart", FontSize=14);
pause
close all