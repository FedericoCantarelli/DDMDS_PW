%identified model using the 
% code in id_test A(z) = 1 - 0.9211 z^-1 + 0.2555 z^-2 - 0.2081 z^-3

clear all
clc

e = randn(10000,1);
y = filter(1,[1.0000   -0.9000    0.2500   -0.2250],e);

yout = y+0.05*[1:10000]';