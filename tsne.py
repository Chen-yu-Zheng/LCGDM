# coding='utf-8'

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import torch
from module.Encoder import Deeplabv2
# from module.Benchmark import DeepLabV3Plus
# from module.Encoder import DenseFusionNet
# from module.csn import DistNorm2d, change_dn
from skimage.io import imread
from albumentations import Compose, Normalize, Resize
import ever as er
# from data.cross_data import reclassify
import numpy as np
from collections import OrderedDict

# color_class = [(255, 255, 255), (255, 0, 0), (255, 255, 0), (0, 0, 255),
#                (159, 129, 183), (0, 255, 0), (255, 195, 128)]

color_class = ['c', 'b', 'g', 'r', 'm', 'y', 'k']

LABEL_MAP = OrderedDict(
    Background=0,
    Building=1,
    Road=2,
    Water=3,
    Barren=4,
    Forest=5,
    Agricultural=6
)

COLOR_MAP_LoveDA = OrderedDict(
    Background=(255, 255, 255),
    Building=(255, 0, 0),
    Road=(255, 255, 0),
    Water=(0, 0, 255),
    Barren=(159, 129, 183),
    Forest=(0, 255, 0),
    Agricultural=(255, 195, 128),
)

def load_AdaptSeg():
    model = Deeplabv2(dict(
        backbone=dict(
            resnet_type='resnet34',
            output_stride=16,
            pretrained=True,
        ),
        multi_layer=True,
        cascade=True,
        use_ppm=False,
        ppm=dict(
            num_classes=4,
            use_aux=False,
            fc_dim=512,
        ),
        inchannels=512,
        num_classes=4,
    ))
    state_dict = torch.load('D:\WorkSpace\LoveCS\log\AdaptSeg.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict, strict=True)
    print('Load AdaptSeg OK!')
    model.eval()
    return model

def load_CBST(path):
    model = Deeplabv2(dict(
        backbone=dict(
                resnet_type='resnet50',
                output_stride=16,
                pretrained=True,
            ),
        multi_layer=False,
        cascade=False,
        use_ppm=True,
        ppm=dict(
            num_classes=7,
            use_aux=False,
        ),
        inchannels=2048,
        num_classes=7
    ))

    state_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    print('Load CBST OK!')
    model.eval()
    return model

def load_CLAN(path):
    model = Deeplabv2(dict(
        backbone=dict(
            resnet_type='resnet50',
            output_stride=16,
            pretrained=True,
        ),
        multi_layer=True,
        cascade=False,
        use_ppm=False,
        ppm=dict(
            num_classes=7,
            use_aux=False,
        ),
        inchannels=2048,
        num_classes=7
    ))

    state_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    print('Load CLAN OK!')
    model.eval()
    return model

def process_image(image_path, model):
    image = imread(image_path)
    transforms=Compose([
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()

    ])
    image = transforms(image=image)['image']
    results = model(image[None, :, :, :]).softmax(dim=1)
    return results

if __name__ == '__main__':
    np.random.seed(2333)
    tar_image_path = "./datasets/LoveDA/Test/Urban/images_png/5167.png"
    tar_mask_path = "./datasets/LoveDA/Test/Urban/masks_png/5167.png"
    
    tar_image = imread(tar_image_path)
    tar_mask = imread(tar_mask_path).astype(np.long) -1

    # path = "./log/cbst/cbst_seed2333_iter20k/2urban/URBAN6000.pth"
    # path='./log/cbst/cbst_20k_ks4_cn7_seed2333/2urban/URBAN8000.pth'
    # path = "./log/cbst/cbst_ent_seed2333/2urban/URBAN8000.pth"
    # model = load_CBST(path)

    path = "./log/clan/20k_seed2333/2urban/URBAN4000.pth"
    # path="./log/clan/20k_ent_e-1/2urban/URBAN4000.pth"
    # path = "./log/clan/20k_ks4_cn7_seed2333_w6k_l1e-1/2urban/URBAN8000.pth"
    model = load_CLAN(path)
    
    tar_results = process_image(tar_image_path, model)[0].detach().numpy().transpose(1, 2, 0)


    sample_per_class = 3000

    tar_res_list = []
    labels = []


    for i in range(7):

        tar_res_i = tar_results[tar_mask == i]
        np.random.shuffle(tar_res_i)
        tar_res_i = tar_res_i[:sample_per_class]
        tar_res_list.append(tar_res_i)
        
        labels += sample_per_class * [i]

    tar_res = np.vstack(tar_res_list)

    embedded_result = tar_res
    print(embedded_result.shape)

    scaler = StandardScaler()
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30)
    embedded_result = scaler.fit_transform(embedded_result)
    embedded_result = tsne.fit_transform(embedded_result)
    
    print(embedded_result.shape)

    pic_path = './clan.png'
    plt.figure()
    # plt.scatter(embedded_result[sample_per_class*i:sample_per_class*(i+1), 0], embedded_result[sample_per_class*i:sample_per_class*(i+1), 1], color=color_class[i], s=1)
    plt.scatter(embedded_result[:,0], embedded_result[:,1], c=labels, s=1, cmap='rainbow')
    plt.savefig(pic_path)
    plt.close()
