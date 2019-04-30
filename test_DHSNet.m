%predict salient object saliency maps.
clear,clc
addpath('caffe/matlab');

%if you don't have a CUDA enabled GPU for acceleration, change
%it to 0, then the code will run slowly.
use_gpu=1;
gpu_id=1;
if use_gpu
  caffe.set_mode_gpu();
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end
model_file='pretrained_models/DHSNet.caffemodel';
model_config='prototxts/deploy-DHSNet.prototxt';

net = caffe.Net(model_config, model_file, 'test');

mean_pix = [103.939, 116.779, 123.68];


imgPath         = 'test-Image/';
resultsPath     = 'results/DHS/';
mkdir(resultsPath);

imgFiles=dir([imgPath '/*.jpg']);
imgNum=length(imgFiles);

t_start         = tic;
for i=1:imgNum
    disp(['Processing the ' num2str(i) 'st image out of ' num2str(imgNum)]);

    image=imread([imgPath '/' imgFiles(i).name]);
    [imgName,~]=strtok(imgFiles(i).name,'.');

    im = single(image);
    im = imresize(im, [224, 224]);
    if size(image,3) == 1
        im = cat(3, im, im, im);
    end
    im = im(:, :, [3 2 1]);
    im = permute(im, [2 1 3]);
    for c = 1:3
        im(:, :, c) = im(:, :, c) - mean_pix(c);
    end

    net.blobs('img').set_data(im);
    net.forward_prefilled();
    
    sm=imresize((net.blobs('RCL1_sm').get_data())',[size(image,1),size(image,2)]);

    imwrite(sm,[resultsPath '/' imgName '.png']);
end
t_end           = toc(t_start);
fps             = round(imgNum/t_end);
fprintf('FPS : %d\n', fps);
caffe.reset_all();