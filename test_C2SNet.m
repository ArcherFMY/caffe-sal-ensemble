% res + fcn
clc;clear;
close all;

%% init
% Add caffe/matlab to my Matlab search PATH to use matcaffe
if exist('caffe/matlab/+caffe','dir')
    addpath('caffe/matlab');
else
    error('Please run this from caffe/matlab/demo');
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
net_model       = 'prototxts/deploy-C2SNet.prototxt';
net_weights     = 'pretrained_models/C2SNet.caffemodel';
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
respath         = 'results/C2SNet/';
if ~exist(respath, 'dir')
    mkdir(respath);
end
im_ext          = '.jpg';
res_ext         = '.png';
imnames         = dir([impath '*' im_ext]);
im_num          = numel(imnames);
input_dim       = 224;
t_start         = tic;
EPSILON = 1e-8;
for i = 1 : im_num
    fprintf('Processing images: %05d/%05d\n', i, im_num);
    im                  = imread([impath, imnames(i).name]);
    [im_data, r, c]     = im_preprocess(im, input_dim, mean_pix, 'input');
    input_data = {imresize(im_data, [input_dim, input_dim])};
    if max(r, c) > 500
        input_data = {imresize(im_data, [input_dim, input_dim])};
        net.blobs('data').reshape([input_dim, input_dim, 3, 1])
    end
    res         = net.forward(input_data);
    out1 = net.blobs('saliencymap').get_data();
    
%     final_map = (out1 - min(out1(:)) + EPSILON) / (max(out1(:)) - min(out1(:)) + EPSILON);

    final_map   = map_postprocess(out1, r, c, 'noneed');

    imwrite(final_map, [respath, imnames(i).name(1:end-4), res_ext]);
end
t_end           = toc(t_start);
fps             = round(im_num/t_end);
fprintf('FPS : %d\n', fps);
caffe.reset_all();