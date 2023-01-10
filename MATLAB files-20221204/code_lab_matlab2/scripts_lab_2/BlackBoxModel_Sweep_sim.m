clc
clear all

%% these data were obtained simulating the "BlackBox_Model_sweep" 
%% Simulink file
load sweep_data

%% Plots

figure
plot(time,input)
hold on
plot(time,output)
legend('Input','Model Output')
xlabel('Time [s]')
grid
ylim([-10 10])

iodata=iddata(output,input,0.01);
freq_res1=tfest(iodata,1,0);
freq_res2=tfest(iodata,2,0);
freq_res3=tfest(iodata,3,0);

n1=freq_res1.Numerator;
d1=freq_res1.Denominator;
est1=tf(n1,d1);

n2=freq_res2.Numerator;
d2=freq_res2.Denominator;
est2=tf(n2,d2);

n3=freq_res3.Numerator;
d3=freq_res3.Denominator;
est3=tf(n3,d3);

real_sis=tf(8,[1 3 3 1]);

figure
bode(est1,est2,est3,real_sis)
legend('est1','est2','est3','real_sis')
grid

pole(est3)
pole(real_sis)

% %compute the state-space representation of the system
% sspace_est3=ss(est3)
% 
% Q1=100*eye(3);
% R1=1;
% 
% Q2=50*eye(3);
% R2=10;
% 
% %discrete time --> dlqr
% [K1,S1,e1] = lqr(sspace_est3,Q1,R1,0) %compute the optimal gain matrix K.
% [K2,S2,e2] = lqr(sspace_est3,Q2,R2,0) %compute the optimal gain matrix K.
% 
% 
% disp('eigenvalues in open loop')
% eig(sspace_est3)
% 
% disp('closed loop eigenvalues with K1')
% e1
% 
% disp('closed loop eigenvalues with K2')
% e2
% 
% % % let's build the controlled systems
% a1k=[sspace_est3.A-sspace_est3.B*K1];
% b1k = sspace_est3.B;
% c1k=sspace_est3.C; 
% d1k=sspace_est3.D;
% controlled_1 = ss(a1k,b1k,c1k,d1k);
% 
% % 
% a2k=[sspace_est3.A-sspace_est3.B*K2];
% b2k = sspace_est3.B;
% c2k=sspace_est3.C;
% d2k=sspace_est3.D;
% controlled_2 = ss(a2k,b2k,c2k,d2k);
% 
% 
% % compute the free motion of the uncontrolled (open loop) and of the 
% % controlled systems from given initial conditions
% x_0=[1; 1; 1];
% [y1,t1,x1]=initial(controlled_1,x_0);
% [y2,t2,x2]=initial(controlled_2,x_0);
% [y_ol,t_ol,x_ol]=initial(sspace_est3,x_0);
% 
% figure
% plot(t1,y1,'b',t2,y2,'r',t_ol,y_ol,'k','linewidth',2);grid on; 
% legend('K1','K2','open loop')
% title('Free motion of the output');
% xlabel('time [s]');
% 
% %plot the state evolution
% figure
% subplot(3,1,1)
% plot(t_ol,x_ol(:,1),':r',t_ol,x_ol(:,2),'--b',t_ol,x_ol(:,3),'.k','linewidth',2);grid on; 
% legend('x1 in open loop','x2 in open loop','x3 in open loop')
% grid
% title('State motion - open loop');
% xlabel('time [s]');
% %
% subplot(3,1,2)
% plot(t1,x1(:,1),':r',t1,x1(:,2),'--b',t1,x1(:,3),'.k','linewidth',2);grid on; 
% legend('x1 with K1','x2 with K1','x3 with K1')
% grid
% title('State motion - K1');
% xlabel('time [s]');
% %
% subplot(3,1,3)
% plot(t2,x2(:,1),':r',t2,x2(:,2),'--b',t2,x2(:,3),'.k','linewidth',2);grid on; 
% legend('x1 with K2','x2 with K2','x3 with K2')
% grid
% title('State motion  - K2');
% xlabel('time [s]');
% 
% 
% 
