function output = computeAIC(target,predicted, na, nc, correction)
%Compute the Akaike Information Criterion of the given data and the given
%model
%   INPUT:
% target: ground truth
% model: model to compute the AIC
% correction: if True, use the correction for small samples
errorsq = (target - predicted).^2;

k = na + nc -2;
if ~correction
    output = 2*k/length(target) + 2*log(mean(errorsq));
else
    aic = 2*k/length(target) + 2*log(mean(errorsq));
    output = aic + (2*k^2+2*k)/(length(target)-k-1);
end