function output = computeFPE(target,predicted, na, nc)
%Compute the Final Prediction Error of the given data and the given
%model
%   INPUT:
% target: ground truth
% predicted: y_hat of the model
% na: length of model.A
% nc: length of model.C
errorsq = (target - predicted).^2;
n = na + nc - 2;
num = length(target) + n;
den = length(target) - n;
output = num/den * mean(errorsq);
end