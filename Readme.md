# Model Globally, Match Locally: Efficient and Robust 3D Object Recognition

> Small python implementation of the paper "Model Globally, Match Locally: Efficient and Robust 3D Object Recognition"
> by Drost. et al (2010) [[paper]](https://campar.in.tum.de/pub/drost2010CVPR/drost2010CVPR.pdf).
>
> The code is not very fast, so bring some time ;)

![](screenshot.png)

# Usage

```
$ python3 ppf.py --help
usage: ppf.py [-h] model scene

This script computes the point-pair features of a given model and tries to find the model in a given scene.

positional arguments:
  model       Path to the model pointcloud
  scene       Path to the scene pointcloud

options:
  -h, --help  show this help message and exit
```

# Gotchas

- This is very much an exploratory implementation to understand the paper. No guarantees for correctness, no speed.
- As `trimesh` currently doesn't support point clouds with normals, input need to be meshes
- Currently, the code does no resampling of the input meshes, so preprocess them
- There are probably some mistakes still in the code, I discover more every now and then

# Changes to Original

- Instead of sampling all other points for a given reference point in the scene,
  I use a KDTree to only look in a certain radius. The radius is the diagonal of
  the model, since further points could not possibly belong to the same object.