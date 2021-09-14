# Panoptic Narrative Grounding

This repository provides a PyTorch implementation of the paper [Panoptic Narrative Grounding](https://arxiv.org/abs/2109.04988). To appear at [ICCV, 2021](https://iccv2021.thecvf.com/) as an Oral Presentation. Panoptic Narrative Grounding is a spatially fine and general formulation of the natural language visual grounding problem. We establish an experimental framework for the study of this new task, including new ground truth and metrics, and we propose a strong baseline method to serve as stepping stone for future work. We exploit the intrinsic semantic richness in an image by including panoptic categories, and we approach visual grounding at a fine-grained level by using segmentations. In terms of ground truth, we propose an algorithm to automatically transfer Localized Narratives annotations to specific regions in the panoptic segmentations of the MS COCO dataset. The proposed baseline achieves a performance of 55.4 absolute Average Recall points. This result is a suitable foundation to push the envelope further in the development of methods for Panoptic Narrative Grounding.
<br/>

## Paper

[Panoptic Narrative Grounding](https://arxiv.org/abs/2109.04988) <br/>
[Cristina González](https://cigonzalez.github.io/)<sup>1</sup>, Nicolás Ayobi<sup>1</sup>, Isabela Hernández<sup>1</sup>, José Hernández <sup>1</sup>, [Jordi Pont-Tuset](https://jponttuset.cat/)<sup>2</sup>, [Pablo Arbeláez](https://scholar.google.com.co/citations?user=k0nZO90AAAAJ&hl=en)<sup>1</sup> <br/>
<sup>1 </sup> Center for Research and Formation in Artificial Intelligence ([CINFONIA](https://cinfonia.uniandes.edu.co/)) , Universidad de Los Andes. <br/>
<sup>2 </sup>Google Research, Switzerland. <br/>

## Installation

### Requirements

- Python
- Numpy
- Pytorch 1.7.1
- Tqdm 4.56.0
- Scipy 1.5.3

### Cloning the repository

```bash
$ git clone git@github.com:BCV-Uniandes/PNG.git
$ cd PNG
```

## Dataset Preparation

### Panoptic Marrative Grounding Benchmark

1. Download the 2017 MSCOCO Dataset from its [official webpage](https://cocodataset.org/#download). You will need the train and validation splits' images1 and panoptic segmentations annotations.

2. Download the Panoptic Narrative Grounding Benchmark and pre-computed features from our [project webpage](https://bcv-uniandes.github.io/panoptic-narrative-grounding/#downloads) with the following folders structure:

```
panoptic_narrative_grounding
|_ images
|  |_ train2017
|  |_ val2017
|_ features
|  |_ train2017
|  |  |_ mask_features
|  |  |_ sem_seg_features
|  |  |_ panoptic_seg_predictions
|  |_ val2017
|     |_ mask_features
|     |_ sem_seg_features
|     |_ panoptic_seg_predictions
|_ annotations
   |_ png_coco_train2017.json
   |_ png_coco_val2017.json
   |_ panoptic_segmentation
      |_ train2017
      |_ val2017
```

### Train setup:

Modify the routes in train_net.sh according to your local paths.

```bash
python main --init_method "tcp://localhost:8080" NUM_GPUS 1 DATA.PATH_TO_DATA_DIR path_to_your_data_dir DATA.PATH_TO_FEATURES_DIR path_to_your_features_dir OUTPUT_DIR output_dir
```

### Test setup:

Modify the routes in test_net.sh according to your local paths.

```bash
python main --init_method "tcp://localhost:8080" NUM_GPUS 1 DATA.PATH_TO_DATA_DIR path_to_your_data_dir DATA.PATH_TO_FEATURES_DIR path_to_your_features_dir OUTPUT_DIR output_dir TRAIN.ENABLE "False"
```

## Pretrained model

To reproduce all our results as reported bellow, you can use our [pretrained model](https://lambda004.uniandes.edu.co/panoptic-narrative-grounding/pretrained-models/model_final.pth) and our source code.

| Method | things + stuff | things | stuff |
| :----: | :------------: | :----: | :---: |
| Oracle |      64.4      |  67.3  | 60.4  |
|  Ours  |      55.4      |  56.2  | 54.3  |
|  MCN   |       -        |  48.2  |   -   |

| Method | singulars + plurals | singulars | plurals |
| :----: | :-----------------: | :-------: | :-----: |
| Oracle |        64.4         |   64.8    |  60.7   |
|  Ours  |        55.4         |   56.2    |  48.8   |

## Citation

If you find Panoptic Narrative Grounding useful in your research, please use the following BibTeX entry for citation:

```
@article{gonzalez2021png,
  title={Panoptic Narrative Grounding},
  author={Gonz{\'a}lez, Cristina and Ayobi, Nicol{'\a}s and Hern{\'a}ndez, Isabela and Hern{\'a}ndez, Jose and Pont-Tuset, Jordi and Arbel{\'a}ez, Pablo},
  journal={arXiv preprint arXiv:2109.04988},
  year={2021},
}
```
