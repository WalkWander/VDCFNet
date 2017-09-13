%% VDCFNet Demo
% Demonstrates VDCFNet on the Jumping sequence of the OTB.
% The reported speed(50 fps) requests GTX 1070 GPU Device 
% and set "param.visual = false" please.
clear; close all; clc;

init_rect = [147,110,34,33];
img_file = dir('./Jumping/img/*.jpg');
img_file = fullfile('./Jumping/img/', {img_file.name});
subS.init_rect = init_rect;
subS.s_frames = img_file;

param = {};
param.net_name = 'VDCFNet-2017-7-20';
param.interp_factor = 0.002;
param.scale_penalty = 0.98;
param.scale_step = 1.03;
param.gpu = true; % false
param.visual = true;% false

tic
res = run_VDCFNet(subS,0,0,param);
disp(['fps: ',num2str(numel(img_file)/toc)]);
