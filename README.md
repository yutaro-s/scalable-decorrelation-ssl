
# Learning Decorrelated Representations Efficiently Using Fast Fourier Transform

This is the official implementation of the following paper:

>Yutaro Shigeto\*, Masashi Shimbo\*, Yuya Yoshikawa, Akikazu Takeuchi. 
Learning Decorrelated Representations Efficiently Using Fast Fourier Transform. 
CVPR 2023.
>
>\* Equal contribution.

[ [arXiv](https://arxiv.org/abs/2301.01569) | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Shigeto_Learning_Decorrelated_Representations_Efficiently_Using_Fast_Fourier_Transform_CVPR_2023_paper.html) | [Short presentation (YouTube)](https://youtu.be/ngPiU13Fg0M) ]


## Setup

1. Clone this repository, including the submodule ([solo-learn](https://github.com/vturrisi/solo-learn))

    ```
    git clone --recurse-submodules https://github.com/yutaro-s/scalable-decorrelation-ssl.git
    ```

2. Build a Docker image

    ```
    make docker-build
    ```

3. Set your API key and username if you intend to use [W&B](https://wandb.ai/)

    ```
    export WANDB_API_KEY=[API key]
    export WANDB_ENTITY=[username]
    ```

4. Launch a Docker container

    ```
    make docker-run
    ```


## Training and Evaluation

1. Self-supervised learning on ImageNet

    ```
    WANDB_PROJECT=[projetc name] bash script/in1k-r50-d8192/pretrain/sbarlow.sh
    ```

2. Linear evaluation on ImageNet

    ```
    WANDB_PROJECT=[projetc name] bash ./script/in1k-r50-d8192/linear/sbarlow.sh [path to the checkpoint]
    ```


## Citation

If you use this code, please cite our paper:

```
@InProceedings{Shigeto_2023_CVPR,
    author    = {Shigeto, Yutaro and Shimbo, Masashi and Yoshikawa, Yuya and Takeuchi, Akikazu},
    title     = {Learning Decorrelated Representations Efficiently Using Fast Fourier Transform},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {2052-2060}
}
```


## Acknowledgments

This repository is built using [solo-learn](https://github.com/vturrisi/solo-learn). I would like to express my gratitude to the authors of solo-learn.

This work is based on results obtained from Project JPNP20006, commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
