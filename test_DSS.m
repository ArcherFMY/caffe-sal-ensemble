% res + fcn
clc;clear;
close all;

%% init
% Add caffe/matlab to my Matlab search PATH to use matcaffe
if exist('caffe/matlab/+caffe','dir')
    addpath('caffe/matlab');
else
    error('Please run this from root');
end
addpath('utils/');
% Set caffe mode
use_gpu     = true;
if exist('use_gpu', 'var') && use_gpu
    caffe.set_mode_gpu();
    gpu_id = 1;
    caffe.set_device(gpu_id);
else
    caffe.set_mode_cpu();
end

% Initialize the network using my caffemodel
net_model       = 'prototxts/deploy-DSS.prototxt';
net_weights     = 'pretrained_models/DSS.caffemodel';
phase           = 'test';
if ~exist(net_weights, 'file')
    error('No such model');
end

% Initialize a network
net         = caffe.Net(net_model, net_weights, phase);
mean_pix    = single([104 117 123]);
mean_pix    = reshape(mean_pix,[1,1,3]);
%% load testing set
impath          = 'test-Image/';
respath         = 'results/DSS/';
% respath_deconv  = ['../res/fcn_stage2_',iter,'/'];
% respath_res     = ['../res/fcn_stage2_',iter,'/'];
if ~exist(respath, 'dir')
    mkdir(respath);
end
im_ext          = '.jpg';
res_ext         = '.png';
imnames         = dir([impath '*' im_ext]);
im_num          = numel(imnames);
input_dim       = 400;
t_start         = tic;
EPSILON = 1e-8;
for i = 1 : im_num
    fprintf('Processing images: %05d/%05d\n', i, im_num);
    im                  = imread([impath, imnames(i).name]);
    [im_data, r, c]     = im_preprocess(im, input_dim, mean_pix, 'input');
%     prior               = imread([priorpath,imnames(i).name(1:end-4),'.png']);
%     [prior_data, ~, ~]  = im_preprocess(prior, prior_dim, mean_pix, 'prior');
%     input_data  = {im_data;prior_data};
    input_data = {imresize(im_data, [c, r])};
    net.blobs('data').reshape([c, r, 3, 1])
    res         = net.forward(input_data);
    out1 = net.blobs('sigmoid-dsn1').get_data();
    out2 = net.blobs('sigmoid-dsn2').get_data();
    out3 = net.blobs('sigmoid-dsn3').get_data();
    out4 = net.blobs('sigmoid-dsn4').get_data();
    out5 = net.blobs('sigmoid-dsn5').get_data();
    out6 = net.blobs('sigmoid-dsn6').get_data();
    fuse = net.blobs('sigmoid-fuse').get_data();
    final_map = (out2 + out3 + out4 + fuse) / 4;

%     final_map = (out3 + out4 + out5 + fuse) / 4;
    final_map = (final_map - min(final_map(:)) + EPSILON) / (max(final_map(:)) - min(final_map(:)) + EPSILON);
    final_map   = map_postprocess(final_map, r, c, 'noneed');

    imwrite(final_map, [respath, imnames(i).name(1:end-4), res_ext]);
end
t_end           = toc(t_start);
fps             = round(im_num/t_end);
fprintf('FPS : %d\n', fps);
caffe.reset_all();