<div align="center">

# 💡**DistillationDPO**💡
## **[Diffusion Distillation With Direct Preference Optimization For Efficient 3D LiDAR Scene Completion](#)**

by An Zhao<sup>1</sup>, *[Shengyuan Zhang](https://github.com/SYZhang0805)<sup>1</sup>, [Ling Yang](https://github.com/YangLing0818)<sup>2</sup>, [Zejian Li*](https://zejianli.github.io/)<sup>1</sup>, Jiale Wu<sup>1</sup>, Haoran Xu<sup>3</sup>, AnYang Wei<sup>3</sup>,Perry Pengyun GU<sup>3</sup>, [Lingyun Sun](https://person.zju.edu.cn/sly)<sup>1</sup>*

*<sup>1</sup>Zhejiang University <sup>2</sup>Peking University <sup>3</sup>Zhejiang Green Zhixing Technology co., ltd*

![](./pics/teaser2.png)

</div>

## **Abstract**

The application of diffusion models in 3D LiDAR scene completion is limited due to diffusion's slow sampling speed. 
Score distillation accelerates diffusion sampling but with performance degradation, while post-training with direct policy optimization (DPO) boosts performance using preference data.
This paper proposes Distillation-DPO, a novel diffusion distillation framework for LiDAR scene completion with preference aligment.
First, the student model generates paired completion scenes with different initial noises.
Second, using LiDAR scene evaluation metrics as preference, we construct winning and losing sample pairs. 
Such construction is reasonable, since most LiDAR scene metrics are informative but non-differentiable to be optimized directly.
Third, Distillation-DPO optimizes the student model by exploiting the difference in score functions between the teacher and student models on the paired completion scenes.
Such procedure is repeated until convergence.
Extensive experiments demonstrate that, compared to state-of-the-art LiDAR scene completion diffusion models, Distillation-DPO achieves higher-quality scene completion while accelerating the completion speed by more than 5-fold.
Our method is the first to explore adopting preference learning in distillation to the best of our knowledge and provide insights into preference-aligned distillation.

## **Environment setup**

The following commands are tested with Python 3.8 and CUDA 11.1.

Install required packages:

`sudo apt install build-essential python3-dev libopenblas-dev`

`pip3 install -r requirements.txt`

Install [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) for sparse tensor processing:

`pip3 install -U MinkowskiEngine==0.5.4 --install-option="--blas=openblas" -v --no-deps`

Setup the code on the code main directory:

`pip3 install -U -e .`

## **Training**

We use The SemanticKITTI dataset for training.

The SemanticKITTI dataset has to be downloaded from the [official site](http://www.semantic-kitti.org/dataset.html#download) and extracted in the following structure:

```
DistillationDPO/
└── datasets/
    └── SemanticKITTI
        └── dataset
          └── sequences
            ├── 00/
            │   ├── velodyne/
            |   |       ├── 000000.bin
            |   |       ├── 000001.bin
            |   |       └── ...
            │   └── labels/
            |       ├── 000000.label
            |       ├── 000001.label
            |       └── ...
            ├── 01/
            │   ...
            ...
```

Ground truth scenes are not provided explicitly in SemanticKITTI. To generate the ground complete scenes you can run the `map_from_scans.py` script. This will use the dataset scans and poses to generate the sequence map to be used as ground truth during training:

```
python map_from_scans.py --path datasets/SemanticKITTI/dataset/sequences/
```

We use the diffusion-dpo fine-tuned version of [LiDiff](https://github.com/PRBonn/LiDiff) as the teacher model as well as the teacher assistant models. Download the pre-trained weights from [here](https://drive.google.com/drive/folders/1z7Iq6nPDZXtASUDP8R8sqhUAvVfRqKQH?usp=sharing) and place it at `checkpoints/lidiff_ddpo_refined.ckpt`.

Once the sequences map is generated and the teacher model is downloaded you can then train the model. The training can be started with:

`python trains/DistillationDPO.py --SemanticKITTI_path datasets/SemanticKITTI --pre_trained_diff_path checkpoints/lidiff_ddpo_refined.ckpt`

## **Inference & Visualization**

We use [pyrender](https://github.com/mmatl/pyrender) for offscreen rendering. Please see [this guide](https://pyrender.readthedocs.io/en/latest/install/index.html#osmesa) for installation of pyrender.

After correct installtion of pyrender, download the refinement model 'refine_net.ckpt' from [here](https://drive.google.com/drive/folders/1z7Iq6nPDZXtASUDP8R8sqhUAvVfRqKQH?usp=sharing) and place it at `checkpoints/refine_net.ckpt`. We also provide the pre-trained weights of distillation-dpo. Download 'distillationdpo_st' from [here](https://drive.google.com/drive/folders/1z7Iq6nPDZXtASUDP8R8sqhUAvVfRqKQH?usp=sharing) and place it at `checkpoints/distillationdpo_st`.

Then run the inference script with the following command:

`python utils/eval_path_get_pics.py --diff checkpoints/distillationdpo_st.ckpt --refine checkpoints/refine_net.ckpt` 

This script will read all scenes in a sepcified sequence of SemanticKITTI dataset and the result images will be saved under `exp/`. 

## **Citation**

If you find our paper useful or relevant to your research, please kindly cite our papers:

```bibtex
TODO
```

## **Credits**

DistillationDPO is highly built on the following amazing open-source projects:

[Lidiff](https://github.com/PRBonn/LiDiff): Scaling Diffusion Models to Real-World 3D LiDAR Scene Completion