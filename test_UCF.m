function  test_UCF()
%################################### Inference Saliency Map ##################################
% This code is used for UCF saliency model 
% Code Author: Pingping Zhang
% Email: jssxzhpp@gmail.com
% Date: 8/8/2017
% The code is based on the following paper in ICCV2017:
% Title: Learning Uncertain Convolutional Features for Accurate Saliency Detection
% Author: Pingping Zhang, Dong Wang, Huchuan Lu, Hongyu Wang and Baocai Yin
%#############################################################################################
%% setting caffe test config
addpath('caffe/matlab/');
use_gpu= 1;
% Set caffe mode
if exist('use_gpu', 'var') && use_gpu
  caffe.set_mode_gpu();
  gpu_id = 1;  % we will use the first gpu in this demo
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end
%% Initialize the network
%###################### ICCV2017 ###########################
%% Amulet
net_model = 'prototxts/deploy-UCF.prototxt';
net_weights = 'pretrained_models/UCF.caffemodel';
phase = 'test'; 
net = caffe.Net(net_model, net_weights, phase);
%% load images from different Datasets
imPath = ['test-Image/'];
salPath = ['results/UCF/'];
if ~exist(salPath, 'dir')
    mkdir(salPath);
end
files = dir([imPath '*.jpg']);
num = length(files);
%% test each image
t_start         = tic;
for i = 1: num
    name = files(i).name(1:end-4);
    im = imread([imPath name '.jpg']);
    if size(im,3)==1
        im = cat(3,im,im,im);
    end
% do forward pass to get scores
    res = net.forward({prepare_image(im)});
%%  if use the prediciton map
   salmap = permute(res{1}(:,:,2), [2 1 3]);
%%  if use the contrast inference
%    for j = 1: length(res)
%       be_map = permute(res{j}(:,:,1), [2 1 3]);
%       fe_map = permute(res{j}(:,:,2), [2 1 3]);
%       diff_map(:,:,j) = fe_map - be_map; 
%    end
%    mean_map = mean(diff_map,3);
%    sal_map = max(0,mean_map);
%% resize and save salmap
    salmap  = imresize(salmap,[size(im,1) size(im,2)], 'bilinear');
     imwrite(salmap, [salPath, name, '.png']);
    fprintf('Processing Img: %d/%d,\n', i, num);
end
   t_end           = toc(t_start);
   fps             = round(num/t_end);
   fprintf('FPS : %d\n', fps);
   caffe.reset_all();
end
   
% ------------------------------------------------------------------------
function images = prepare_image(im)
% ------------------------------------------------------------------------
%IMAGE_DIM = 256;
%IMAGE_DIM = 288;
IMAGE_DIM = 448;
% resize to fixed input size
im = single(im);
im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
% permute from RGB to BGR (IMAGE_MEAN is already BGR)
im = im(:,:,[3 2 1]);
% subtract mean_data (already in W x H x C, BGR)
im(:,:,1) = im(:,:,1) -104 ;  
im(:,:,2) = im(:,:,2) -117 ;
im(:,:,3) = im(:,:,3) -123 ;
images = permute(im,[2 1 3]);
% ------------------------------------------------------------------------
end