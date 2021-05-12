# 3D Object Detection and Semantic Map Generation

### Installation steps
The necessary packages have been enlisted in `requirements.txt` but installation of certain packages like `torch` would have OS-specific instructions.

An outline of the [installation steps](https://github.com/facebookresearch/votenet#installation) are mentioned in VoteNet's repository.

The steps:

- Install the latest Pytorch for your platform
- `pip install -r requirements.txt`
- `cd data/votenet/pointnet2`
- `python3 setup.py install`

**Please note**: The compilation of pointnet2 requires a cuda-enabled gpu.

### Results
The benchbot simulation has very high system spec requirements as listed on [their page](3D Object Detection and Semantic Map Generation). A video recording of an entire run, including the inferences at every frame and the final score, has been shared for convenience in evaluation.

The code we used to interface with the benchbot simulator and generate semantic maps can be found in the [3d semantic map](code/3d_semantic_map) directory.
The three approaches are as follows:

- Votenet pretrained weights (on an incomplete subset of objects)
- Group-free-3D pretrained weights
- Ensemble learning by stacking the results of votenet pretrained weights as well as our own trained model on the remaining subset of objects

An interesting task was to stitch the inferences for every frame. We transformed the inferences to global coordinates (true camera pose is provided)and implemented 3D Non-maximum suppression. This removes the repetitions of the object and allows to get rid of noisy inferences too. The NMS is also performed across classes of objects sharing spatial overlap.

### References
- [How to read an image from diode dataset](https://github.com/diode-dataset/diode-devkit/blob/master/diode.py)
- [Conversion from depth image to pointcloud](https://medium.com/yodayoda/from-depth-map-to-point-cloud-7473721d3f)
- [Using Votenet's pretrained weight](https://github.com/charlesq34/votenet-1/blob/master/demo.py)
- [Pytorch on Colab fix](https://github.com/facebookresearch/votenet/issues/97)
- [SunRGBD v1 vs v2](https://github.com/facebookresearch/votenet/issues/12)
- [Bad results with other SUN-RGBD objects](https://github.com/facebookresearch/votenet/issues/101)
- [Boxnet getting better results than Votenet](https://github.com/facebookresearch/votenet/issues/33)
- [ImportError: Could not import _ext module.](https://github.com/facebookresearch/votenet/issues/108#issuecomment-783878066)

### Resources
- [mAP and other terms](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)
- [Votenet repo](https://github.com/facebookresearch/votenet) and [paper](https://arxiv.org/pdf/1904.09664.pdf)
- [Group Free 3D repo](https://github.com/zeliu98/Group-Free-3D) and [paper](https://arxiv.org/abs/2104.00678)
