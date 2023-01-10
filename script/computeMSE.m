function output = computeMSE(target,predicted)
%Compute the Mean Squared Error of the given data and the given
%model
%   INPUT:
% target: ground truth
% predicted: y_hat of the model
errorsq = (target - predicted).^2;
output = mean(errorsq);
end