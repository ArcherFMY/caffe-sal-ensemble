[![License](https://img.shields.io/badge/License-MIT%20LICENSE-brightgreen.svg)](LICENSE)
# caffe-sal-ensemble

### Introduction
This repository is an ensemble version of several salient object detection repos.

- [x] [AFNet](https://github.com/ArcherFMY/AFNet)
- [x] [Amulet](https://github.com/Pchank/caffe-sal)
- [x] [C2SNet](https://github.com/lixin666/C2SNet)
- [x] DHSNet
- [x] [DSS](https://github.com/Andrew-Qibin/DSS)
- [x] [PiCANet](https://github.com/nian-liu/PiCANet)
- [x] [RAS](https://github.com/ShuhanChen/RAS_ECCV18)
- [x] [SRM](https://github.com/Pchank/caffe-sal)
- [x] [UCF](https://github.com/Pchank/caffe-sal)

### Installation
The code is built on Ubuntu 14.04 with CUDA 8.0 and OpenCV 3.1.

### Usage

1.  Clone the repository: 
	```shell
	git clone https://github.com/ArcherFMY/caffe-sal-ensemble.git
	```

2.  Build Caffe and matcaffe:
	```shell
	cd $CAFFE_SAL_ENSEMBLE_ROOT && cd caffe
	cp Makefile.config.example Makefile.config 
	vim Makefile.config #edit Makefile.config for your environment
	make all -j
	make matcaffe
	```
3.  Run the test demo:
	- The trained model is put in folder `pretrained_models`: -> `xxx.caffemodel` (download their released pretrained models and change the name to match their method name.)
	
	- The network model is put in folder `prototxts`: -> `deploy-xxx.prototxt`
	
	- The test images can be found in `test-Image`
		 
	- Then run `test_xxx.m` using matlab, you can get the saliency maps in folder `results/xxx/`

### Performance

You could use [this evaluation tool box](https://github.com/ArcherFMY/sal_eval_toolbox).
	
### License
This code is released under the MIT License (refer to the LICENSE file for details).

### Contact

Please contact mengyang_feng@mail.dlut.edu.cn
