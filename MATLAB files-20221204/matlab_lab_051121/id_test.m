%let's perform the identification and prediction on the given dataset

%first step: upload the data
clear; load data051121


%let's plot the original data: we clearly see a linear trend
N = length(yout);
figure(1); plot(yout);
xlabel('Time')
ylabel('y(t)')
legend('original data')

pause

%pre-process the data to remove the linear trend
y = detrend(yout);				%% de-trend of signal z
tr = yout-y;				%% trend which has been removed
%plot the trend and the detrended data
figure(2); plot(y); hold on; plot(tr,'r');
legend('detrended data','trend')
xlabel('Time')
ylabel('y(t)')
pause

%let's compute and plot the PSD of the detrended process
w = [-pi:2*pi/10000:pi-1/10000];		%% frequencies axis
G = fftshift(abs(fft(y)).^2);		%% spectrum computed via FFT
figure(3); plot(w,G);
xlabel('Frequency')
ylabel('PSD')

%and the covariance function
figure(4); plot(covf(y,150),'*');		%% covariance computed via covf
grid
xlabel('\tau')
ylabel('covariance')
pause


%% now we estimate an autoregressive model for the data, computing also the loss function and three different cross-correlation
%% indexes, the FPE, AIC and MDL
for i=1:10
    th{i} = arx(iddata(y),i);			%% arx identification
    J(i) = th{i}.EstimationInfo.LossFcn;	%% loss function
    V1(i) = fpe(th{i});				%% FPE
    V2(i) = aic(th{i});				%% AIC
    V3(i) = mdl(th{i});				%% MDL
end

%%let's plot the cost function and the three cross-correlation indexes
figure(5); hold on; plot(J,'r')
 hold on; plot(V1,'b');
 plot(V2,'g');plot(V3,'k');
 legend('J','FPE','AIC','MDL')
pause

%%we choose the model order as that minimizing the MDL criterion
[a b] = min(V3)				%% model chosen by minimizing MDL
yhat = predict(th{b},y);			%% let's evaluate the prediction capability of the chosen model
figure(6); hold on;
plot(y); plot(yhat,'r');
legend('y','y_{hat}')

%%let's compare now the original and predicted data by adding back the
%%trend
figure(7); hold on;
plot(y); plot(yhat+tr,'r');
legend('original data','predicted global output')
pause

%%let's compute the prediction error
res=y-yhat;
figure(8)
plot(res)
legend('prediction error')

%%let's evaluate the optimality of the model class, by checking whether the
%%residual is white or not
figure(9)
resid(y,th{b});				%% whiteness test