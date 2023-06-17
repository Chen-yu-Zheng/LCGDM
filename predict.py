from ast import arg
from data.loveda import LoveDALoader
from utils.tools import *
from skimage.io import imsave
import os
import argparse

from albumentations import Normalize
from albumentations import Compose
import ever as er

import warnings
warnings.filterwarnings("ignore")


def predict_test(model, cfg, ckpt_path, save_dir='./submit_test'):
    os.makedirs(save_dir, exist_ok=True)
    seed_torch(42)
    model_state_dict = torch.load(ckpt_path)
    model.load_state_dict(model_state_dict,  strict=True)

    count_model_parameters(model)

    model.eval()
    print(cfg.TEST_DATA_CONFIG)
    eval_dataloader = LoveDALoader(cfg.TEST_DATA_CONFIG)

    with torch.no_grad():
        for ret, ret_gt in tqdm(eval_dataloader):
            ret = ret.to(torch.device('cuda'))
            cls = model(ret)
            cls = cls.argmax(dim=1).cpu().numpy()
            for fname, pred in zip(ret_gt['fname'], cls):
                imsave(os.path.join(save_dir, fname), pred.astype(np.uint8))

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CBST methods.')
    parser.add_argument('--model_path',  type=str, default='./log/cbst_sl_30k_ks10_cn10_seed42/2urban/URBAN30000.pth', help='model path')
    parser.add_argument('--urban', action='store_true', default=False, help='if 2urban')
    parser.add_argument('--save_path',  type=str, default='./submit_test', help='save pred path')
    args = parser.parse_args()

    from module.Encoder import Deeplabv2
    if args.urban:
        cfg = import_config('st.cbst.2urban')
    else:
        cfg = import_config('st.cbst.2rural')

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
            num_classes=cfg.NUM_CLASSES,
            use_aux=False,
        ),
        inchannels=2048,
        num_classes=cfg.NUM_CLASSES
    )).cuda()

    predict_test(model, cfg, args.model_path, save_dir=args.save_path)