clc;clear;close all;

addpath(genpath('caffe/matlab'));

inpath = ['test-Image/'];
outpath = ['results/SRM/'];
if ~exist(outpath, 'dir')
    mkdir(outpath);
end

in_dir=dir([inpath, '*.jpg']);
imgnum=length(in_dir);
% 
net_model='prototxts/deploy-SRM.prototxt';
net_weights='pretrained_models/SRM.caffemodel';
phase='test';
caffe.set_mode_gpu();
caffe.set_device(0);
% Initialize a network
net=caffe.Net(net_model, net_weights, phase);

IMAGE_DIM = 353; % set input to 353*353
t_start         = tic;
for i=1:imgnum
    i
    imgname=in_dir(i).name;
    img_name=[inpath imgname];
    out_name=[outpath imgname(1:end-4) '.png'];
    
    im=imread(img_name);
    im = single(im);
    [m n k]=size(im);
    % resize to fixed input size
    if k == 1
        im = cat(3, im, im, im);
    end
    im = imresize(im, [IMAGE_DIM IMAGE_DIM]);

    % RGB -> BGR
    im = im(:, :, [3 2 1]);

    images = zeros(IMAGE_DIM, IMAGE_DIM, 3, 1, 'single');
    images(:,:,1:3,1)=permute(im,[2 1 3]);

    % No mean subtraction is required for the input image. 
    % There is a batch-normalization layer which basically does the same, as described in reference [41].
    % [41] M.Simon, E.Rodner, and J.Denzler. Imagenet pretrained models with batch normalization. arXiv preprint arXiv:1612.01452,2016

    for c = 1:3
    %         images(:, :, c, :) = images(:, :, c, :) - mean_pix(c);
        images(:, :, c, :) = images(:, :, c, :) ;
    end
    input_data = {images};

    % do forward pass to get scores
    scores=net.forward(input_data);

    temp=scores{1}(:,:,2);
    map=permute(temp,[2,1,3]); 

    map=imresize(map,[m,n]);
    % normalize to get final saliency map
    map=(map-min(map(:)))./(max(map(:))-min(map(:)));
    imwrite(map,out_name,'png');

end
t_end           = toc(t_start);
fps             = round(imgnum/t_end);
fprintf('FPS : %d\n', fps);
caffe.reset_all();
