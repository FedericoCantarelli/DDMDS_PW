function output = computeMAE(target, predicted)
%Compute the Mean Absolute Error of the given data and the given
%model
%   INPUT:
% target: ground truth
% predicted: y_hat of the model
error = target - predicted;
output = mean(abs(error));
end