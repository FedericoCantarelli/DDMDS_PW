%% Control chart
close all
clear all
clc

%% Load environment
load("best_model_found.mat");

%% Create residuals
optimal_model = model_min_MDL;
unbiased_transition = transition_activity - M;
unbiased_earthquake = earthquake_activity - M;

t_train = 1:960;
t_test = 961:1200;
t_trans = 1201:1700;
t_earth = 1701:3048;


y_hat = predict(my_data_train, optimal_model, 1).y;
e_train = my_data_train.y - y_hat;

y_hat = predict(my_data_test, optimal_model, 1).y;
e_test = my_data_test.y - y_hat;

y_hat = predict(iddata(unbiased_transition, TS=1), optimal_model, 1).y;
e_trans = unbiased_transition - y_hat;

y_hat = predict(iddata(unbiased_earthquake, Ts=1), optimal_model, 1).y;
e_earth = unbiased_earthquake - y_hat;



%% Random selection

idx = randperm(length(e_train));
e_train = e_train(idx(1:10:end));
t_train = t_train(idx(1:10:end));

idx = randperm(length(e_test));
e_test = e_test(idx(1:10:end));
t_test = t_test(idx(1:10:end));

idx = randperm(length(e_trans));
e_trans = e_trans(idx(1:10:end));
t_trans = t_trans(idx(1:10:end));

idx = randperm(length(e_earth));
e_earth = e_earth(idx(1:10:end));
t_earth = t_earth(idx(1:10:end));

%%
mat_train = [t_train.' e_train];
mat_train = sortrows(mat_train, 1);

mat_test = [t_test.' e_test];
mat_test = sortrows(mat_test, 1);

mat_trans = [t_trans.' e_trans];
mat_trans = sortrows(mat_trans, 1);

mat_earth = [t_earth.' e_earth];
mat_earth = sortrows(mat_earth, 1);

%%
e_train = mat_train(:,2);
e_test = mat_test(:,2);
e_trans = mat_trans(:,2);
e_earth = mat_earth(:,2);

t_train = 1:96;
t_test = 97:120;
t_trans = 121:170;
t_earth = 171:305;

plot(t_train, e_train, t_test, e_test, t_trans, e_trans, t_earth, e_earth)



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
plot(t_train, e_train, t_test, e_test, t_trans, e_trans, t_earth, e_earth)
hold on
yline(I_CL, "--g")
hold on
yline(I_LCL, "--r")
hold on
yline(I_UCL, "--r")
hold on
title("I-chart on batched residuals", FontSize=14)
pause
close all

% Remove the 10th observation
e_train_no_10 = e_train;
e_train_no_10(10) = [];

mr = zeros(length(e_train_no_10)-1,1);
for i=1:length(e_train_no_10)-1
    mr(i)=abs(e_train_no_10(i+1)-e_train_no_10(i));
end

I_CL = mean(e_train_no_10);
I_LCL = I_CL - z/1.128 * mean(mr);
I_UCL = I_CL + z/1.128 * mean(mr);


figure
plot(1:95, e_train_no_10, 96:119, e_test, 120:169, e_trans, 170:304, e_earth)
hold on
yline(I_CL, "--g")
hold on
yline(I_LCL, "--r")
hold on
yline(I_UCL, "--r")
hold on
title("I-chart on batched residuals", "10th observation excluded", FontSize=14)
pause
close all

%% MR chart
MR_CL = mean(mr);
MR_LCL = 0;
MR_UCL = 3.267*MR_CL;

data = [e_test; e_trans; e_earth];
mr_2 = zeros(length(data)-1,1);
for i=1:length(data)-1
    mr_2(i)=abs(data(i+1)-data(i));
end

% Split for visualisation purposes
mr_test = mr_2(1:24);
mr_trans = mr_2(25:74);
mr_earth = mr_2(75:208);

figure
plot(1:94, mr, 95:118, mr_test, 119:168, mr_trans, 169:302, mr_earth)
hold on
yline(MR_CL, "--g")
hold on
yline(MR_LCL, "--r")
hold on
yline(MR_UCL, "--r")
xlim([0 302])
pause
close all

%% EWMA
lambda = 0.05;
L = 2.615;

mu = mean(e_train);
sigma = sqrt(var(e_train));

data = [e_train; e_test; e_trans; e_earth];

EWMA = cell(1+length(data));
EWMA{1} = mu;

for i=2:length(data)
    EWMA{i} = lambda*data(i) + (1-lambda)*EWMA{i-1};
end

EWMA = cell2mat(EWMA);

CL = mu;
UCL = cell(length(data));
LCL = cell(length(data));

for idx=1:length(UCL)
    k = lambda/(2-lambda)*(1-(1-lambda)^(2*idx));
    UCL{idx} = CL + L*sigma*sqrt(k);
    LCL{idx} = CL - L*sigma*sqrt(k);
end

% split for viz purpose
ewma_train = EWMA(1:96);
ewma_test = EWMA(97:120);
ewma_trans = EWMA(121:170);
ewma_earth = EWMA(171:305);


figure
plot(cell2mat(UCL), "r--", LineWidth=1.2);
hold on 
plot(cell2mat(LCL), "r--", LineWidth=1.2);
hold on
yline(mu, "g--", LineWidth=1.2);
hold on
plot(1:96, ewma_train, 97:120, ewma_test, 121:170, ewma_trans, 171:305, ewma_earth);
hold on
xlim([1 305]);
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

MA(1) = data(1);
MA(2) = mean(data(1:2));
MA(3) = mean(data(1:3));
MA(4) = mean(data(1:4));

UCL(1) = mu+3*sigma/sqrt(1);
UCL(2) = mu+3*sigma/sqrt(2);
UCL(3) = mu+3*sigma/sqrt(3);
UCL(4) = mu+3*sigma/sqrt(4);

LCL(1) = mu-3*sigma/sqrt(1);
LCL(2) = mu-3*sigma/sqrt(2);
LCL(3) = mu-3*sigma/sqrt(3);
LCL(4) = mu-3*sigma/sqrt(4);

for i=w:length(data)
    MA(i)=mean(data(i-4:i));
end

UCL(5:end)=mu+3*sigma/sqrt(5);
LCL(5:end)=mu-3*sigma/sqrt(5);

% Split for viz
ma_train = MA(1:96);
ma_test = MA(97:120);
ma_trans = MA(121:170);
ma_earth = MA(171:305);
plot(1:96, ma_train, 97:120, ma_test, 121:170, ma_trans, 171:305, ma_earth)
hold on
plot(LCL);
hold on 
plot(UCL);
hold on
plot(MA);
pause
close all