import os
import time
import torch
import torchvision as tv
import GS_Jacobi_sampling
import utils
import pathlib
from torch.utils.data import Dataset, DataLoader
from PIL import Image
utils.set_random_seed(19)
import numpy as np
from collections import defaultdict
import random
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

####################################################################################################
## Initialize the TarFlow model setting 

# This paper contains four models
dataset = 'imagenet'
# dataset = 'imagenet64_patch4'
# dataset = 'afhq'
# dataset = 'imagenet64_uncond'

dataset1 = {'imagenet': 'imagenet', 'imagenet64_uncond': 'imagenet64','imagenet64_cond': 'imagenet', 'afhq': 'afhq','imagenet64_patch4':'imagenet'}[dataset]
num_classes = {'imagenet': 1000, 'imagenet64_uncond': 0,'imagenet64_cond': 1000, 'afhq': 3, 'imagenet64_patch4': 1000}[dataset]
img_size = {'imagenet': 128, 'imagenet64_uncond': 64, 'imagenet64_cond': 64, 'afhq': 256, 'imagenet64_patch4': 64}[dataset]
patch_size = {'imagenet': 4, 'imagenet64_uncond': 2,'imagenet64_cond': 2, 'afhq': 8, 'imagenet64_patch4': 4}[dataset]
channels = {'imagenet': 1024, 'imagenet64_uncond': 768, 'imagenet64_cond': 768,'afhq': 768, 'imagenet64_patch4': 1024}[dataset]
noise_std = {'imagenet': 0.15, 'imagenet64_uncond': 0.05,'imagenet64_cond': 0.05, 'afhq': 0.07, 'imagenet64_patch4': 0.05}[dataset]
channel_size = 3
blocks = 8
layers_per_block = 8

device = 'cuda:7'

# The paths of data and model parameters
data_path = '/mnt/iem-nas/home/qinzhen/qinzhen/experiment_generation/tar_flow/data'
if dataset == 'afhq':
    ckpt_file = '/mnt/iem-nas/home/qinzhen/qinzhen/experiment_generation/tar_flow/runs/cond_afhq_256/afhq_model_8_768_8_8_0.07.pth'
elif dataset == 'imagenet':
    ckpt_file = '/mnt/iem-nas/home/qinzhen/qinzhen/experiment_generation/tar_flow/runs/cond_imagenet_128/imagenet_model_4_1024_8_8_0.15.pth'
elif dataset == 'imagenet64_uncond':
    ckpt_file = '/mnt/iem-nas/home/qinzhen/qinzhen/experiment_generation/tar_flow/runs/uncond_imagenet_64/imagenet64_model_2_768_8_8_0.05.pth'
elif dataset == 'imagenet64_cond':
    ckpt_file = '/mnt/iem-nas/home/qinzhen/qinzhen/experiment_generation/tar_flow/runs/cond_imagenet_64/imagenet_model_2_768_8_8_0.05.pth'
elif dataset == 'imagenet64_patch4':
    ckpt_file ='/mnt/iem-nas/home/qinzhen/qinzhen/experiment_generation/tar_flow/runs/cond_imagenet_64_patch_4_300_ep/imagenet_model_4_1024_8_8_0.05.pth'

####################################################################################################
## Calculate IGN and CRN

# IGN and CRN are calculated in the forward process of TarFlow, which need to pass a small batch of 
# training set through the models
batch_size = 1000
# set detect_mode = True, set the norm (1,2,'fro')
norm = 2
model = GS_Jacobi_sampling.Model(in_channels=channel_size, img_size=img_size, patch_size=patch_size, 
              channels=channels, num_blocks=blocks, layers_per_block=layers_per_block,
             num_classes=num_classes,detect_mode=True,norm=norm).to(device)
model.load_state_dict(torch.load(ckpt_file))
print('checkpoint loaded!')

num_mc = 1000
class RandomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=1000):
        self.root_dir = root_dir
        self.transform = transform
        self.max_samples = max_samples
        
        self.all_files = self._find_image_files(root_dir)
        self.sampled_files = random.sample(self.all_files, min(max_samples, len(self.all_files)))

    def _find_image_files(self, directory):
        image_extensions = ('.jpg', '.png', '.jpeg', '.JPEG')
        image_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_files.append(os.path.join(root, file))
        return image_files

    def __len__(self):
        return len(self.sampled_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.sampled_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

transform = tv.transforms.Compose([
    tv.transforms.Resize(img_size),
    tv.transforms.CenterCrop(img_size),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5,), (0.5,)),
])

start_time = time.time()
dataset = RandomImageDataset(
    root_dir=os.path.join(data_path, dataset1),
    transform=transform,
    max_samples=num_mc
)
print(f"Total images in subset: {len(dataset)}")

data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,     
    num_workers=8,
    pin_memory=True,
)
print(f"Data loading cost: {time.time() - start_time:.2f}秒")

images = next(iter(data_loader))
images = images.to(device)
y = None
with torch.no_grad():
    z, outputs, logdets, IGN_list, CRN_list = model(images,y) 

print(IGN_list)
print(CRN_list)
CRN_percent = [crn[3] for crn in CRN_list]
CRN_percent = np.array(CRN_percent)/sum(CRN_percent)*100
print(CRN_percent)

###################################################################################################
# Image Transform Trace

sample_dir = pathlib.Path(f'./NIPS/Trace/')
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

# # Original trace don't need any setting

# # Img64cond
# num_GS_list = [1] * 8
# max_Jacobi_list = [20] * 8

# # Trace without initial guessing
# max_Jacobi_list = [30] * 8
# guess_list = [0,0,0,0,0,0,0,0]

# # Pure Jacobi Trace iterate 20 times
# max_Jacobi_list = [20] * 8
# guess_list = [1,0,0,0,0,1,1,1]

# # Pure Jacobi Trace iterate 30 times
# max_Jacobi_list = [30] * 8
# guess_list = [1,0,0,0,0,1,1,1]

# # GS-Jacobi sampling
num_GS_list = [8,1,1,1,1,1,1,1]
max_Jacobi_list = [13,6,6,6,6,6,6,20]
guess_list = [1,0,0,0,0,1,1,1]

guidance = 1.5
attn= 1.0
with torch.no_grad():
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        # # Original trace
        # samples = model.reverse(fixed_noise, fixed_y, guidance=guidance, attn_temp=attn,annealed_guidance=True,return_sequence=True)
        samples = model.reverse(fixed_noise, fixed_y, guidance=guidance, attn_temp=attn,annealed_guidance=True, return_sequence=True,
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
save_path = sample_dir/ f"Img64cond_trace.png"
tv.utils.save_image(grid, save_path)

# # Original Trace
# input_paths = [
#     "Img128cond_trace.png",  
#     "Img64cond_trace.png",
#     "Img64uncond_trace.png",  
#     "AFHQ_trace.png"   
# ]

input_paths = [
    "Img64cond_noIGN.png",  
    "Img64cond_jacobi20.png",  
    "Img64cond_jacobi30.png",  
    "Img64cond_GSJ.png",  
]

resized_images = []
for path in input_paths:
    img = Image.open(sample_dir/ path)
    target_height = 128  
    target_width = 1280  
    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    resized_images.append(img)

final_image = Image.new("RGB", (1280, 512))
y_offset = 0
for img in resized_images:
    final_image.paste(img, (0, y_offset))
    y_offset += img.height
output_path = "GS-Jacobi_trace.pdf"
final_image.save(sample_dir/ output_path, quality=95)

####################################################################################################
# Converge Curve for Pure Jacobi Iteration for all blocks

sample_dir = pathlib.Path(f'./NIPS/Trace/')
sample_dir.mkdir(exist_ok=True, parents=True)
batch_size = 128
fixed_noise = torch.randn(batch_size, (img_size // patch_size)**2, channel_size * patch_size ** 2, device=device)
if num_classes:
    fixed_y = torch.randint(num_classes, (batch_size,), device=device)
else:
    fixed_y = None

model = GS_Jacobi_sampling.Model(in_channels=channel_size, img_size=img_size, patch_size=patch_size, 
              channels=channels, num_blocks=blocks, layers_per_block=layers_per_block,
             num_classes=num_classes).to(device)
model.load_state_dict(torch.load(ckpt_file))
print('checkpoint loaded!')

# Sampling with original "For" loop to get the target X
guidance = 2.0
attn = 1.0
with torch.no_grad():
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        samples = model.reverse(fixed_noise, fixed_y, guidance=guidance, attn_temp=attn,annealed_guidance=True,return_sequence=True)
print(f'{dataset1} sampling completed')
samples = [model.patchify(trace) for trace in samples]
X_target_list = samples[1:]


num_GS_list = [1] * blocks
max_Jacobi_list = [1023] * blocks
# # For Img64cond, max Jacobi times is 255
# max_Jacobi_list = [255] * blocks

# guess_list = [0,1,0,0,1,0,1,1]
# guess_list = [0,1,0,0,0,0,1,0]
# guess_list = [1,0,0,0,0,1,1,1]
# guess_list = [0,0,0,0,0,1,1,0]

with torch.no_grad():
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        output, trace_list = model.reverse(fixed_noise, fixed_y, guidance=guidance, attn_temp=attn,annealed_guidance=True,
                                           num_GS_list=num_GS_list,max_jacobi_list=max_Jacobi_list,
                                           guess_list=guess_list,show_trace=True,X_target_list=X_target_list)

# Save the trace
with open(sample_dir/'img64cond_tracefull.pkl', 'wb') as f:
    pickle.dump(trace_list, f)

# Load the traces and draw
sns.set_theme()
with open(sample_dir/'img64cond_tracefull.pkl', 'rb') as f:
    trace_list_img64cond = pickle.load(f)
with open(sample_dir/'img64uncond_tracefull.pkl', 'rb') as f:
    trace_list_img64uncond = pickle.load(f)
with open(sample_dir/'img128tracefull.pkl', 'rb') as f:
    trace_list_img128 = pickle.load(f)
with open(sample_dir/'afhq_tracefull.pkl', 'rb') as f:
    trace_list_afhq = pickle.load(f)

fig = plt.figure(figsize=(16, 9))
x = np.arange(1, 151)  # Only show the first 150 iterations
colors = plt.cm.viridis(np.linspace(0, 0.8, 8))
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':'] 
blocks = 8 

titles = [
    'Img128cond',
    'AFHQ',
    'Img64cond',
    'Img64uncond'
]

trace_lists = [
    trace_list_img128,
    trace_list_afhq,
    trace_list_img64cond,
    trace_list_img64uncond,
]

plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
for idx, (trace_list, title, scale) in enumerate(zip(trace_lists, titles)):
    ax = plt.subplot(2, 2, idx+1)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    for i in range(blocks):
        ax.plot(x, np.abs(np.array(trace_list[i][:150])-trace_list[i][150]), 
                label=f'block{7-i}',
                color=colors[i],
                linewidth=3.0,
                alpha=0.8,
                linestyle=linestyles[i])
    
    ax.set_title(title, fontsize=20)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis="both", which="major", labelsize=20)
    legend = ax.legend(fontsize=18, bbox_to_anchor=(1.0, 1), loc='upper right')
    legend.get_frame().set_boxstyle("Round", pad=0.2)

plt.tight_layout()
plt.savefig(sample_dir/'combined_trace_plots.png', dpi=300, bbox_inches='tight')
plt.show()

####################################################################################################
# Converge Curve for GS-Jacobi iteration

sample_dir = pathlib.Path(f'./NIPS/GSJ_Trace/')
sample_dir.mkdir(exist_ok=True, parents=True)
batch_size = 128

fixed_noise = torch.randn(batch_size, (img_size // patch_size)**2, channel_size * patch_size ** 2, device=device)
if num_classes:
    fixed_y = torch.randint(num_classes, (batch_size,), device=device)
else:
    fixed_y = None

model = GS_Jacobi_sampling.Model(in_channels=channel_size, img_size=img_size, patch_size=patch_size, 
              channels=channels, num_blocks=blocks, layers_per_block=layers_per_block,
             num_classes=num_classes).to(device)
model.load_state_dict(torch.load(ckpt_file))

guidance = 2.0
attn = 1.0
with torch.no_grad():
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        samples = model.reverse(fixed_noise, fixed_y, guidance=guidance, attn_temp=attn,annealed_guidance=True,return_sequence=True)
print(f'{dataset1} sampling completed')
samples = [model.patchify(trace) for trace in samples]

X_target_list = samples[1:]

# img64cond
num_GS_list = [0,0,0,0,0,0,0,8]
max_Jacobi_list = [1,1,1,1,1,1,1,32]
guess_list = [1,0,0,0,0,1,1,1]

# # # img64uncond 
# # num_GS_list = [0,8,0,0,0,0,0,0]
# # max_Jacobi_list = [1,128,1,1,1,1,1,1]
# # guess_list = [0,1,0,0,1,0,1,1]

# # # img128cond
# # num_GS_list = [0,8,0,0,0,0,0,0]
# # max_Jacobi_list = [1,128,1,1,1,1,1,1]
# # guess_list = [0,1,0,0,0,0,1,0]

# # AFHQ
# num_GS_list = [8,0,0,0,0,0,0,0]
# max_Jacobi_list = [128,1,1,1,1,1,1,1]
# guess_list = [0,0,0,0,0,1,1,0]

with torch.no_grad():
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        output, trace_list = model.reverse(fixed_noise, fixed_y, guidance=guidance, attn_temp=attn,annealed_guidance=True,
                                           num_GS_list=num_GS_list,max_jacobi_list=max_Jacobi_list,
                                           guess_list=guess_list,show_trace=True,X_target_list=X_target_list)

# Save the trace
with open(sample_dir/'img64cond_tracefull.pkl', 'wb') as f:
    pickle.dump(trace_list, f)

# Load the traces and draw
with open(sample_dir/'img64cond_tracefull.pkl', 'rb') as f:
    trace_list_img64cond = pickle.load(f)
trace_list_img64cond = trace_list_img64cond[7]

with open(sample_dir/'img64uncond_tracefull.pkl', 'rb') as f:
    trace_list_img64uncond = pickle.load(f)
trace_list_img64uncond = trace_list_img64uncond[1]

with open(sample_dir/'img128cond_tracefull.pkl', 'rb') as f:
    trace_list_img128 = pickle.load(f)
trace_list_img128 = trace_list_img128[1]

with open(sample_dir/'afhq_tracefull.pkl', 'rb') as f:
    trace_list_afhq = pickle.load(f)
trace_list_afhq = trace_list_afhq[0]

fig = plt.figure(figsize=(16, 9))
colors = plt.cm.viridis(np.linspace(0, 0.8, 8)) 
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':'] 
modules = 8 

titles = [
    'Img128cond Block6',
    'AFHQ Block7',
    'Img64cond Block0',
    'Img64uncond Block6'
]

trace_lists = [
    trace_list_img128,
    trace_list_afhq,
    trace_list_img64cond,
    trace_list_img64uncond,
]


plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15

for idx, (trace_list, title) in enumerate(zip(trace_lists, titles)):
    ax = plt.subplot(2, 2, idx+1)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0),useMathText=True)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    
    for i in range(modules):
        ax.plot(np.arange(1, len(trace_list[i])+1)[:32], (np.abs(np.array(trace_list[i])-trace_list[i][-1]))[:32]/8, 
                label=f'modules{i+1}',
                color=colors[i],
                linewidth=3.0,
                alpha=0.8,
                linestyle=linestyles[i])
    
    ax.set_title(title, fontsize=30)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis="both", which="major", labelsize=20)

    legend = ax.legend(fontsize=18, bbox_to_anchor=(1.0, 1), loc='upper right')
    legend.get_frame().set_boxstyle("Round", pad=0.2) 

plt.tight_layout()
plt.savefig(sample_dir/'combined_GSJ_trace_plots.png', dpi=300, bbox_inches='tight')
plt.show()


####################################################################################################
# Compare samples generated by For and GS-Jacobi

sample_dir = pathlib.Path(f'./NIPS/samples/')
sample_dir.mkdir(exist_ok=True, parents=True)
batch_size = 25
fixed_noise = torch.randn(batch_size, (img_size // patch_size)**2, channel_size * patch_size ** 2, device=device)
if num_classes:
    fixed_y = torch.randint(num_classes, (batch_size,), device=device)
else:
    fixed_y = None

model = GS_Jacobi_sampling.Model(in_channels=channel_size, img_size=img_size, patch_size=patch_size, 
              channels=channels, num_blocks=blocks, layers_per_block=layers_per_block,
             num_classes=num_classes).to(device)
model.load_state_dict(torch.load(ckpt_file))
print('checkpoint loaded!')

# AFHQ
guidance = 1.0
attn= 1.0
num_GS_list = [0,1,1,1,1,1,1,1]
max_Jacobi_list = [32,10,10,10,10,10,10,10]
guess_list = [0,0,0,0,0,1,1,0]

# # img128cond
# guidance = 1.5
# attn= 1.0
# num_GS_list = [1,8,1,1,1,1,1,1]
# max_Jacobi_list = [10,32,10,10,10,10,10,10]
# guess_list = [0,1,0,0,0,0,1,0]

# # Img64cond
# guidance = 2.0
# attn= 1.0
# num_GS_list = [8,1,1,1,1,1,1,0]
# max_Jacobi_list = [13,6,6,6,6,6,6,0]
# guess_list = [1,0,0,0,0,1,1,1]

with torch.no_grad():
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        # # Original "For" loop
        # samples = model.reverse(fixed_noise, fixed_y, guidance=guidance, attn_temp=attn,annealed_guidance=True,return_sequence=False)
        samples = model.reverse(fixed_noise, fixed_y, guidance=guidance, attn_temp=attn,annealed_guidance=True,return_sequence=False,
                                num_GS_list=num_GS_list,max_jacobi_list=max_Jacobi_list,guess_list=guess_list)
print(f'{dataset1} sampling completed')

for p in model.parameters():
    p.requires_grad = False

lr = batch_size * img_size ** 2 * channel_size * noise_std ** 2
x = torch.clone(samples).detach()
x.requires_grad = True
with torch.autocast(device_type=device, dtype=torch.bfloat16):
    z, outputs, logdets = model(x, fixed_y)
loss = model.get_loss(z, logdets)
grad = torch.autograd.grad(loss, [x])[0]
x.data = x.data - lr * grad
samples = x
print(f'{dataset1} denoising complete')
tv.utils.save_image(samples, sample_dir / f'afhq_gsj.png', normalize=True, nrow=5)
