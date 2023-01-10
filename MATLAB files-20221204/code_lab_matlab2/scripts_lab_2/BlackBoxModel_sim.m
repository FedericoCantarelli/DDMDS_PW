%%% LABORATORIO IDENTIFICAZIONE %%%
%%% Alessandro Amodio %%%

clear all
close all
clc

%% Parameters

Ts = 1e-3;                             

step_time = 1;                          

step_amplitude = 1;

sim_time = 20;

%% Simulation 

sim BlackBox_Model.slx

%% Plots
time=0:Ts:sim_time;
figure
plot(time,input)
hold on
plot(time,output)
legend('Input','Model Output')
xlabel('Time [s]')
grid



%% TF Identification

iodelay = finddelay(input,output)*Ts;
datainout = iddata(output,input,Ts);
n_poles = 3;
n_zeros = 0;
TF = tfest(datainout,n_poles,n_zeros,iodelay);

num = TF.Numerator;
den = TF.Denominator;

G2 = tf(num,den)

s = tf('s');
G1 = 1*exp(-0.8*s)/(1+3.7*s);

%% Plot

output_G1 = lsim(G1,input,time);
output_G2 = lsim(G2,input,time);

figure
plot(time,input)
hold on
plot(time,output)
xlabel('Time [s]')
grid
plot(time,output_G1,'g')
plot(time,output_G2,'c')
legend('Input','Model Output','G_1 Output','G_2 Output')
xlim([0 sim_time])

pause

step_amplitude = step_amplitude*10; 
old_output=output;

sim BlackBox_Model.slx
output_10=output;

figure
plot(time,old_output*10,'b','linewidth', 2)
hold on
plot(time,output_10,'r-')
legend('A = 1 (*10)','A = 10')
xlabel('Time [s]')
grid
