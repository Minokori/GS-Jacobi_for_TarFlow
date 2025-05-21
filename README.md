# Accelerate TarFlow Sampling with GS-Jacobi Iteration
![GSJprocess_00](https://github.com/user-attachments/assets/137e66de-ca14-4f21-947a-58fc57a8c27f)

## Transfomer AutoRegressive Flow Model
The TarFlow model is proposed in [[Zhai et al., 2024]](http://arxiv.org/abs/2412.06329), associated code can be found at https://github.com/apple/ml-tarflow.
It achieves state-of-the-art results across multiple benchmarks compared to previous normalizing-flow-based model such as Glow. Its sampling trace looks like this:
![combined_trace](https://github.com/user-attachments/assets/dbde8b88-d7fd-4bac-b8a3-b6eb30ebb47c)
From top to bottom: Img128cond, Img64cond (patch4), Img64uncond, AFHQ. From left to right: noise, Block 7-0, denoised image.

The model parameters are not available in original paper, so we retrain TarFlow models and upload them. As metioned in [[Zhai et al., 2024]](http://arxiv.org/abs/2412.06329), a TarFlow model can be denoted as P-Ch-T-K-pε. The pretrained models used in experiments can be download through:
 - AFHQ (256x256) conditional: [afhq_model_8_768_8_8_0.07.pth](https://huggingface.co/encoreus/Transformer_Autoregressive_Flow/blob/main/afhq_model_8_768_8_8_0.07.pth)
 - ImageNet (64x64) unconditional: [imagenet64_model_2_768_8_8_0.05.pth](https://huggingface.co/encoreus/Transformer_Autoregressive_Flow/blob/main/imagenet64_model_2_768_8_8_0.05.pth)
 - ImageNet (64x64) conditional: [imagenet_model_4_1024_8_8_0.05.pth](https://huggingface.co/encoreus/Transformer_Autoregressive_Flow/blob/main/imagenet_model_4_1024_8_8_0.05.pth)
 - ImageNet (128x128) conditional: [imagenet_model_4_1024_8_8_0.15.pth](https://huggingface.co/encoreus/Transformer_Autoregressive_Flow/blob/main/imagenet_model_4_1024_8_8_0.15.pth)

Download them into `./models/`. Meanwhile, we also compute the stats for the true data distribution which can be used to calculate FID.
 - AFHQ (256x256) conditional: [afhq_256_fid_stats.pth](https://huggingface.co/encoreus/Transformer_Autoregressive_Flow/blob/main/afhq_256_fid_stats.pth)
 - ImageNet (128x128) conditional: [imagenet_128_fid_stats.pth](https://huggingface.co/encoreus/Transformer_Autoregressive_Flow/blob/main/imagenet_128_fid_stats.pth)
 - ImageNet (64x64) unconditional: [imagenet_128_fid_stats.pth](https://huggingface.co/encoreus/Transformer_Autoregressive_Flow/blob/main/imagenet_128_fid_stats.pth)
 - ImageNet (64x64) conditional: [imagenet_64_fid_stats.pth](https://huggingface.co/encoreus/Transformer_Autoregressive_Flow/blob/main/imagenet_64_fid_stats.pth)

Download them into `./data/`, which can avoid re-calculated the FID stats across millions of images.

## Gauss–Seidel-Jacobi Sampling for TarFlow

The sampling process of TarFlow is extremely slow, and we accelerate it in [Accelerate TarFlow Sampling with GS-Jacobi Iteration](https://arxiv.org/abs/2505.12849). we discover that blocks in the TarFlow model have varying importance: a small number of blocks play a major role in image generation tasks, while other blocks contribute relatively little; some blocks are sensitive to initial values and prone to numerical overflow, while others are relatively robust. The simple process is shown in the top figure. 

### Calculate the IGM and CRM for each Blocks
Forward pass a small batch of images in training set, to calculate two metric: IGM and CRM. 
 - IGM (Initial Guessing Metric)  is used to evaluate whether the initial value of the iteration is appropriate;
 - CRM (Convergence Ranking Metric) is used to identify whether a TarFlow block is "simple" (converges in few iterations) or "tough" (requires more iterations).

The `./data/` folder should contain `imagenet/` `imagenet64/` `afhq/` three dataset, which can be downloaded through:
- [Imagenet](https://www.image-net.org/download.php)
- [Imagenet64](https://arxiv.org/abs/1601.06759)
- [AFHQ](https://www.kaggle.com/datasets/dimensi0n/afhq-512)
  
Indeed, if you don't want to retrained the models, or re-calculated the FID stas (use the fid_stats we prepared), you can only contain a very small size of training set, like 1000 images in each folder. The following code will recursively sample `num_mc` images and calculate IGM, CRM:

```bash
python IGM_CRM.py --dataset=imagenet --norm=2 --num_mc=1000
python IGM_CRM.py --dataset=afhq --norm=2 --num_mc=1000
python IGM_CRM.py --dataset=imagenet64_patch4 --norm=2 --num_mc=1000
python IGM_CRM.py --dataset=imagenet64_uncond --norm=2 --num_mc=1000
```

The CRM of four models are:

|Block| Img128cond      | AFHQ      | Img64uncond      | Img64cond |
|------------|------------|------------|------------|------------|
| Block0   | 5.22  |  6.33  | **50.71** |**74.58**|
| Block1   | 5.63  | 6.28   | 2.42 | 4.88 |
| Block2   |  2.47 | 8.16   |2.29  | 0.72 |
| Block3   |  10.93 | 7.94   | 3.38 | 0.96 |
| Block4   | 7.74  |  9.01  | 1.77 | 4.05 |
| Block5   |  7.35 |  10.27  | 1.33 | 2.68 |
| Block6   |  **56.57** | 9.36   | **33.62** | 1.62 |
| Block7   | 4.05   |  **42.60**  |4.44| **10.46** |

For non-dominant blocks, we just need to allocate very few Jacobi iterations times, for domonant "tough" blocks, we should segment it into so-called "GS modules", iteration in parallel with modules, then serial deliver the solution to the next module.

### Evaluate FID with Different GS-Jacobi Strategies
The GS-Jacobi strategies can be denoted as **[Stack-GS-J-Else]**. **Stack** indicates the tough blocks should be segmented; **GS** indicates the number of equal size segmentation with length $`T//GS`$; **J** indicates the maximum Jacobi times of each module; **Else** indicates the maximum Jacobi times for other blocks with small CRM. 

In coding, we present the strategies in three list: `num_GS_list` `max_jacobi_list` `guess_list`. For example, for 8 blocks model, strategy **[6-8-32-10]** with all zero initialization is: 
```
--num_GS_list=[1,8,1,1,1,1,1,1]
--max_jacobi_list=[10,32,10,10,10,10,10,10]
--guess_list=[1,1,1,1,1,1,1,1]
```
and the original "For" loop sampling is set by `num_GS_list=[0,0,0,0,0,0,0,0]`. Our best GSsampling GS-Jacobi sampling for four models are:
```bash
# AFHQ (256x256) conditional
torchrun --standalone --nproc_per_node=8 evaluate_fid.py --dataset=afhq\
  --logdir=runs_GSJ/imagenet128 --ckpt_file=models/afhq_model_8_768_8_8_0.07.pth\
  --img_size=256 --patch_size=8 --channels=768 --noise_std=0.07\
  --cfg=3.4 --self_denoising_lr=1.4\
  --sample_name='afhqcond[7-8-32-10]'\
  --num_GS_list=[8,1,1,1,1,1,1,1]\
  --max_jacobi_list=[32,10,10,10,10,10,10,10]\
  --guess_list=[0,0,0,0,0,1,1,0]

# ImageNet (128x128) conditional
torchrun --standalone --nproc_per_node=8 evaluate_fid.py --dataset=imagenet\
  --logdir=runs_GSJ/afhq --ckpt_file=models/imagenet_model_4_1024_8_8_0.15.pth\
  --img_size=128 --patch_size=4 --channels=1024 --noise_std=0.15\
  --cfg=1.5 --self_denoising_lr=0.97\
  --sample_name='img128cond[6-8-32-10]'\
  --num_GS_list=[1,8,1,1,1,1,1,1]\
  --max_jacobi_list=[10,32,10,10,10,10,10,10]\
  --guess_list=[0,1,0,0,0,0,1,0]

# ImageNet (64x64) unconditional
torchrun --standalone --nproc_per_node=8 evaluate_fid.py --dataset=imagenet64\
  --logdir=runs_GSJ/img64uncond --ckpt_file=models/imagenet64_model_2_768_8_8_0.05.pth\
  --img_size=64 --patch_size=2 --channels=768 --noise_std=0.05\
  --cfg=0.2 --self_denoising_lr=0.9 --attn_temp=0.3\
  --sample_name='img64uncond[0/6-8-32-20]'\
  --num_GS_list=[1,8,1,1,1,1,1,8]\
  --max_jacobi_list=[20,32,20,20,20,20,20,32]\
  --guess_list=[0,1,0,0,1,0,1,1]

# ImageNet (64x64) conditional
torchrun --standalone --nproc_per_node=8 evaluate_fid.py --dataset=imagenet\
  --logdir=runs_GSJ/img64cond --ckpt_file=models/imagenet_model_4_1024_8_8_0.05.pth\
  --img_size=64 --patch_size=4 --channels=1024 --noise_std=0.05\
  --cfg=2.0 --self_denoising_lr=1.0\
  --sample_name='img64cond[0/7-256/8-1/13-6]'\
  --num_GS_list=[8,1,1,1,1,1,1,0]\
  --max_jacobi_list=[13,6,6,6,6,6,6,1]\
  --guess_list=[1,0,0,0,0,1,1,1]
```
The best hyper-parameters `cfg` `self_denoisong_lr` `attn_temp` may vary among machines. On 8 A800 GPUS, The speed up results are:
| | Img128cond      | AFHQ      | Img64uncond      | Img64cond |
|------------|------------|------------|------------|------------|
|"For" time|133.19|109.24|109.05|12.16|
|"For" FID|5.06|13.60 |14.67|4.42|
|Strategy|[6-8-32-10]|[7-8-32-10]|[0/6-8-32-20]|[0/7-256/8-1/13-6]|
|time|29.41|20.54|36.84|4.97|
|FID|5.09|13.73 |14.80|4.42|
|Rate|4.53|5.32|2.96|2.51|

A visual comparation between the original "For" sampling and GS-Jacobi sampling in AFHQ model is:
| ![afhq_ori](https://github.com/user-attachments/assets/c4730870-c4ec-47be-925d-3dbbd0df855e)| ![afhq_gsj](https://github.com/user-attachments/assets/286c78b6-d804-4757-ab7d-99da9ea77854)|
|:------------------:|:------------------:|
|Original Sampling     | GS-Jacobi Sampling       |


### Generate Sampling Trace
Following code can generate sampling trace for each sampling strategy:
```
python Sampling.py --dataset=afhq --cfg=2 \
  --num_GS_list=[8,1,1,1,1,1,1,1]\
  --max_jacobi_list=[32,10,10,10,10,10,10,10]\
  --guess_list=[0,0,0,0,0,1,1,0]
```
The result is like:
![AFHQ_trace](https://github.com/user-attachments/assets/0de206fb-e29c-43fe-8315-0b19ae1707e0)
From left to right: noise, Block 7-0, denoised image.


## BibTeX
```bibtex
@misc{liu2025gsjtarflow,
      title={Accelerate TarFlow Sampling with GS-Jacobi Iteration}, 
      author={Ben Liu and Zhen Qin},
      year={2025},
      eprint={2505.12849},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.12849}, 
}
```
