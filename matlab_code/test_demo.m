%********************
% Correspondence-Free Non-Rigid Point Set Registration Using Unsupervised
% Clustering Analysis, CVPR 2024   Mingyang Zhao & Jingen Jiang
% Initialized on May 27th, 2024, Hong Kong
%********************
close all;
clc;
clear; 


addpath('./src');
addpath('./utils/');


% File root
src="../data/tr_reg_059.ply";
tgt="../data/tr_reg_057.ply";


src=pcread(src);
tgt=pcread(tgt);


% Normalize the point sets
src_pt=src.Location;
tgt_pt=tgt.Location;


[src_pt_normal,src_pre_normal]=data_normalize_input(src_pt);
[tgt_pt_normal,tgt_pre_normal]=data_normalize_input(tgt_pt);



% Downsample point sets make their size less than 5,000
src_pt_normal=pointCloud(src_pt_normal);
tgt_pt_normal=pointCloud(tgt_pt_normal);

% gridStep=0.03; 
% 
% src_pt_normal=pcdownsample(src_pt_normal,'gridAverage',gridStep); 
% tgt_pt_normal=pcdownsample(tgt_pt_normal,'gridAverage',gridStep);

src_pt_normal=double(src_pt_normal.Location);
tgt_pt_normal=double(tgt_pt_normal.Location);

% Show the normalized source and target point clouds
figure;
subplot(1,2,1)
scatter3(src_pt_normal(:,1),src_pt_normal(:,2),src_pt_normal(:,3),'filled');
title("source")
subplot(1,2,2)
scatter3(tgt_pt_normal(:,1),tgt_pt_normal(:,2),tgt_pt_normal(:,3),'filled');
title("target")
hold off;


src_pt_normal_gpu=gpuArray(src_pt_normal);
tgt_pt_normal_gpu=gpuArray(tgt_pt_normal);
[alpha,T_deformed]=fuzzy_cluster_reg(src_pt_normal,tgt_pt_normal);


% Denormalize the deformed point cloud
T_deformed_denormal=denormalize(tgt_pre_normal,T_deformed);
tgt_pt_denormal=denormalize(tgt_pre_normal,tgt_pt_normal);


% Show the original source and target point clouds
figure;
subplot(1,2,1)
scatter3(src_pt(:,1),src_pt(:,2),src_pt(:,3),'filled');
title("source")
subplot(1,2,2)
scatter3(tgt_pt(:,1),tgt_pt(:,2),tgt_pt(:,3),'filled');
title("target")
hold off;


% Show the target and deformed point clouds
figure;
hold on;
scatter3(tgt_pt_denormal(:,1),tgt_pt_denormal(:,2),tgt_pt_denormal(:,3),'filled');
scatter3(T_deformed_denormal(:,1),T_deformed_denormal(:,2),T_deformed_denormal(:,3),'filled');
title("Registration")
hold off;




