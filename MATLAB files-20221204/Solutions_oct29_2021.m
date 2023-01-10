% Exercises

disp('Solutions to Lab Exercises')

%% Linear systems

TrF=tf([1 1],[1 -0.8],-1);

ns=100; % number of samples
[y,T]=step(TrF,ns);

figure
plot(T,y,'k')
step(TrF,ns)

T=[0:ns];
u1=sin(0.1*T);
u2=sin(T);

y1=lsim(TrF,u1,T);
y2=lsim(TrF,u2,T);

figure
plot(T,y1,'r',T,y2,'b')
legend('y_1','y_2')



%% Identification of time series

ns=1000;
e=randn(ns+1,1);
for t=2:ns+1
    y(t)=e(t)+0.5*e(t-1);
end

% ns=10000;
% e2=randn(ns+1,1);
% for t=2:ns+1
%     y2(t)=e2(t)+0.5*e2(t-1);
% end



% model M: y(t)=theta*y(t-1)+e(t)
theta_hat=0.441; % identified using ident (realization-dependent)
theta_bar=(0.5)/(1+0.5^2) % ideal value theta_bar=gamma_y(1)/gamma_y(0);

% optimal predictor: yP(t)=theta*y(t-1) using the AR(1) model
yP=zeros(ns,1);
for t=2:ns
    yP(t)=theta_hat*y(t-1);
    epsilon(t)=y(t)-yP(t);
end
mean_epsilon=mean(epsilon');
covariance_epsilon=covf(epsilon'-mean_epsilon,11);
figure
title('Covariance of \epsilon')
hold on
plot(0:10,covariance_epsilon,'ko')

% theoretical value of the variance:
% var(epsilon)=(1+theta^2)*gamma(0)-2*theta*gamma(1);
var_epsilon_ideal=(1+theta_bar^2)*(1+0.5^2)-2*theta_bar*0.5

%% Model identification

ns=10000;
e=randn(ns,1);
u=4*randn(ns,1); % for identification purposes, the power of the input 
                   % signal must be higher than the power of the noise
y=zeros(ns,1);
for t=3:ns
    y(t)=0.1*y(t-1)+2*u(t-1)-0.8*u(t-2)+e(t)+0.3*e(t-1);
end

% validation data

ns=1000;
ev=randn(ns,1);
uv=4*randn(ns,1); % for identification purposes, the power of the input 
                   % signal is higher than the power of the noise
yv=zeros(ns,1);
for t=3:ns
    yv(t)=0.1*yv(t-1)+2*uv(t-1)-0.8*uv(t-2)+ev(t)+0.3*ev(t-1);
end

% use the identification toolbox for identification, type
% "systemIdentification" (new command name for Ident)


