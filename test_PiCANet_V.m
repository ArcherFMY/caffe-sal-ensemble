%predict salient object saliency maps of a dataset
clear,clc
addpath('caffe/matlab');
% addpath('/home/nianliu/Research/PiCANet_Saliency/caffe/matlab');

caffe.reset_all();
use_gpu=1;
gpu_id=1;
if use_gpu
  caffe.set_mode_gpu();
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end

model_config= 'prototxts/deploy-PiCANet-V.prototxt';
model_file= 'pretrained_models/PiCANet-V.caffemodel';

net = caffe.Net(model_config, model_file, 'test');

mean_pix = [104.008, 116.669, 122.675];

%scale={'5','4','3','2','1'};
scale={'1'};

impath          = 'test-Image/';
respath         = 'results/PiCANet-V';

mkdir(respath);

im_ext          = '.jpg';
res_ext         = '.png';
imnames         = dir([impath '*' im_ext]);
im_num          = numel(imnames);
input_dim       = 224;
t_start         = tic;

for i=1:im_num
    disp(i)

    image                  = imread([impath, imnames(i).name]);

    im = single(image);
    im = imresize(im, [224, 224]);
    if size(im,3)==1
        im = cat(3, im, im, im);
    end
    im = im(:, :, [3 2 1]);
    im = permute(im, [2 1 3]);
    for c = 1:3
        im(:, :, c) = im(:, :, c) - mean_pix(c);
    end

    net.blobs('img').set_data(im);
    net.forward_prefilled();

    for j=1:length(scale)
        sm=imresize((net.blobs(['sm_' scale{j}]).get_data())',[size(image,1),size(image,2)]);
        imwrite(sm, [respath, '/', imnames(i).name(1:end-4), res_ext]);
    end
end
t_end           = toc(t_start);
fps             = round(im_num/t_end);
fprintf('FPS : %d\n', fps);
caffe.reset_all();