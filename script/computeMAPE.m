function output = computeMAPE(target,predicted)
%Compute the mean absolute percentage error of the given data and the given
%model
%   INPUT:
% target: ground truth
% model: model to compute the MAPE
error = target - predicted;
output = mean(abs(error)./target) * 100;
end