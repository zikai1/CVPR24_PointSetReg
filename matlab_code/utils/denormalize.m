function [data_denormal]=denormalize(data_normalized,data_deformed)
%=========================
% Denormalize the input normalized data
% INPUT:
%=========================
%      data_normalized     structure  [data_normalized.xscale,...
%                                      data_normalized.xd]
%      data_transformed    MxN array  the transformed data
% OUTPUT:
%=========================
%      data_denormal       MxN array  the denormalized data
%=========================

M=size(data_normalized,1);

% The translation and scale of tgt
data_denormal=data_deformed*data_normalized.xscale+repmat(data_normalized.xd,M,1);
