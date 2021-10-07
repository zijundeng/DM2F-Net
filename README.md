# DM2F-Net

By Zijun Deng, Lei Zhu, Xiaowei Hu, Chi-Wing Fu, Xuemiao Xu, Qing Zhang, Jing Qin, and Pheng-Ann Heng.

This repo is the implementation of
"[Deep Multi-Model Fusion for Single-Image Dehazing](https://openaccess.thecvf.com/content_ICCV_2019/papers/Deng_Deep_Multi-Model_Fusion_for_Single-Image_Dehazing_ICCV_2019_paper.pdf)"
(ICCV 2019), written by Zijun Deng at the South China University of Technology.

## Results

The dehazing results can be found at 
[Google Drive](https://drive.google.com/drive/folders/1ZVBI_3Y2NthVLeK7ODMIB5vRjmN9payF?usp=sharing).

## Installation & Preparation

Make sure you have `Python>=3.7` installed on your machine.

**Environment setup:**

1. Create conda environment

       conda create -n dm2f
       conda activate dm2f

2. Install dependencies (test with PyTorch 1.8.0):

   1. Install pytorch==1.8.0 torchvision==0.9.0 (via conda, recommend).

   2. Install other dependencies

          pip install -r requirements.txt

* Prepare the dataset

   * Download the RESIDE dataset from the [official webpage](https://sites.google.com/site/boyilics/website-builder/reside).

   * Download the O-Haze dataset from the [official webpage](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/).

   * Make a directory `./data` and create a symbolic link for uncompressed data, e.g., `./data/RESIDE`.

## Training

1. ~~Set the path of pretrained ResNeXt model in resnext/config.py~~
2. Set the path of datasets in tools/config.py
3. Run by ```python train.py```

~~The pretrained ResNeXt model is ported from the [official](https://github.com/facebookresearch/ResNeXt) torch version,
using the [convertor](https://github.com/clcarwin/convert_torch_to_pytorch) provided by clcarwin. 
You can directly [download](https://drive.google.com/open?id=1dnH-IHwmu9xFPlyndqI6MfF4LvH6JKNQ) the pretrained model ported by me.~~

Use pretrained ResNeXt (resnext101_32x8d) from torchvision.

*Hyper-parameters* of training were set at the top of *train.py*, and you can conveniently
change them as you need.

Training a model on a single ~~GTX 1080Ti~~ TITAN RTX GPU takes about ~~4~~ 5 hours.

## Testing

1. Set the path of five benchmark datasets in tools/config.py.
2. Put the trained model in `./ckpt/`.
2. Run by ```python test.py```

*Settings* of testing were set at the top of `test.py`, and you can conveniently
change them as you need.

## License

DM2F-Net is released under the [MIT license](LICENSE).

## Citation

If you find the paper or the code helpful to your research, please cite the project.

```
@inproceedings{deng2019deep,
  title={Deep multi-model fusion for single-image dehazing},
  author={Deng, Zijun and Zhu, Lei and Hu, Xiaowei and Fu, Chi-Wing and Xu, Xuemiao and Zhang, Qing and Qin, Jing and Heng, Pheng-Ann},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={2453--2462},
  year={2019}
}
```
