function output = computeMDL(target,predicted, na, nc)
%Compute the Minimum Description Lenght of the given data and the given
%model
%   INPUT:
% target: ground truth
% predicted: y_hat of the model
% na: length of model.A
% nc: length of model.C
errorsq = (target - predicted).^2;
output = log(length(target))*(na + nc - 2)/length(target)+log(mean(errorsq));
end