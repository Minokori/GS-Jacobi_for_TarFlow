import argparse
import builtins
import pathlib

import ast
import os
import torch
import torchvision as tv
import GS_Jacobi_sampling
import utils
utils.set_random_seed(19)

def main(args):
    dataset = args.dataset
    dataset1 = {'imagenet': 'imagenet', 'imagenet64_uncond': 'imagenet64','imagenet64_cond': 'imagenet', 'afhq': 'afhq','imagenet64_patch4':'imagenet'}[dataset]
    num_classes = {'imagenet': 1000, 'imagenet64_uncond': 0,'imagenet64_cond': 1000, 'afhq': 3, 'imagenet64_patch4': 1000}[dataset]
    img_size = {'imagenet': 128, 'imagenet64_uncond': 64, 'imagenet64_cond': 64, 'afhq': 256, 'imagenet64_patch4': 64}[dataset]
    patch_size = {'imagenet': 4, 'imagenet64_uncond': 2,'imagenet64_cond': 2, 'afhq': 8, 'imagenet64_patch4': 4}[dataset]
    channels = {'imagenet': 1024, 'imagenet64_uncond': 768, 'imagenet64_cond': 768,'afhq': 768, 'imagenet64_patch4': 1024}[dataset]
    noise_std = {'imagenet': 0.15, 'imagenet64_uncond': 0.05,'imagenet64_cond': 0.05, 'afhq': 0.07, 'imagenet64_patch4': 0.05}[dataset]
    channel_size = 3
    blocks = 8
    layers_per_block = 8

    device = 'cuda:0'

    # The paths of data and model parameters
    data_path = './data'
    if dataset == 'afhq':
        ckpt_file = './models/afhq_model_8_768_8_8_0.07.pth'
    elif dataset == 'imagenet':
        ckpt_file = './models/imagenet_model_4_1024_8_8_0.15.pth'
    elif dataset == 'imagenet64_uncond':
        ckpt_file = './models/imagenet64_model_2_768_8_8_0.05.pth'
    elif dataset == 'imagenet64_cond':
        ckpt_file = './models/imagenet_model_2_768_8_8_0.05.pth'
    elif dataset == 'imagenet64_patch4':
        ckpt_file ='./models/imagenet_model_4_1024_8_8_0.05.pth'

    device = 'cuda:0'

    ###################################################################################################
    # Image Transform Trace
    sample_dir = pathlib.Path(f'./Sampling_trace/')
    sample_dir.mkdir(exist_ok=True, parents=True)
    # Generate only one image
    batch_size = 1
    fixed_noise = torch.randn(batch_size, (img_size // patch_size)**2, channel_size * patch_size ** 2, device=device)

    # Random setting the classes
    if num_classes:
        fixed_y = torch.randint(num_classes, (batch_size,), device=device)
    else:
        fixed_y = None
    # # For Imagenet, we generate no.333 (hamster)
    # fixed_y = torch.ones((batch_size,),dtype=int, device=device) * 333

    model = GS_Jacobi_sampling.Model(in_channels=channel_size, img_size=img_size, patch_size=patch_size, 
                channels=channels, num_blocks=blocks, layers_per_block=layers_per_block,
                num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(ckpt_file))
    print('checkpoint loaded!')

    # # GS-Jacobi sampling
    num_GS_list = args.num_GS_list
    max_Jacobi_list = args.max_jacobi_list
    guess_list = args.guess_list

    guidance = args.cfg

    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            samples = model.reverse(fixed_noise, fixed_y, guidance=guidance,annealed_guidance=True, return_sequence=True,
                                               num_GS_list=num_GS_list,max_jacobi_list=max_Jacobi_list,
                                               guess_list=guess_list,)
    final_img = samples[blocks]
    print(f'{dataset1} sampling completed')

    for p in model.parameters():
        p.requires_grad = False
    lr = batch_size * img_size ** 2 * channel_size * noise_std ** 2
    x = torch.clone(final_img).detach()
    x.requires_grad = True
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        z, outputs, logdets = model(x, fixed_y)
    loss = model.get_loss(z, logdets)
    grad = torch.autograd.grad(loss, [x])[0]
    x.data = x.data - lr * grad
    final_denoise = x
    print(f'{dataset1} denoising complete')

    def normalize_per_image(images):
        normalized = []
        for img in images:
            img_min = img.min()
            img_max = img.max()
            img_normalized = (img - img_min) / (img_max - img_min + 1e-8)
            normalized.append(img_normalized)
        return normalized

    all_images = samples + [final_denoise]
    all_images_normalized = normalize_per_image(all_images)
    all_images_tensor = torch.cat(all_images_normalized, dim=0) 
    grid = tv.utils.make_grid(all_images_tensor, nrow=10, padding=0, normalize=False)
    save_path = sample_dir/ f"sampling_trace.png"
    tv.utils.save_image(grid, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='imagenet', type=str, choices=['imagenet', 'imagenet64_patch4', 'afhq', 'imagenet64_uncond'], help='Name of dataset')
    parser.add_argument('--cfg', default=0, type=float, help='Guidance weight for sampling, 0 is no guidance. For conditional models consider the range in [1, 3]')
    parser.add_argument('--num_GS_list',default=[0,0,0,0,0,0,0,0], type=ast.literal_eval, help='the num of GS for each flow blocks, 0 for pure iteration')
    parser.add_argument('--max_jacobi_list', default=[0,0,0,0,0,0,0,0], type=ast.literal_eval, help='the max num of iteration for each flow blocks')
    parser.add_argument('--guess_list', default=[0,0,0,0,0,0,0,0], type=ast.literal_eval, help='initialize x with zero or z')  
    args = parser.parse_args()
    main(args)