from PIL import Image
import numpy as np
import os, glob
from skimage.io import imread


class VisualizeSegmm(object):
    def __init__(self, out_dir, palette):
        self.out_dir = out_dir
        self.palette = palette
        os.makedirs(self.out_dir, exist_ok=True)

    def __call__(self, y_pred, filename):
        """
        Args:
            y_pred: 2-D or 3-D array of shape [1 (optional), H, W]
            filename: str
        Returns:
        """
        y_pred = y_pred.astype(np.uint8)
        y_pred = y_pred.squeeze()
        color_y = Image.fromarray(y_pred)
        color_y.putpalette(self.palette)
        color_y.save(os.path.join(self.out_dir, filename))

def render_path(mask_path, vis_path, palette):
    # new_mask = np.array(Image.open(mask_path)).astype(np.uint8)
    # cm = np.array(list(COLOR_MAP.values())).astype(np.uint8)
    # color_img = cm[new_mask]
    # color_img = Image.fromarray(np.uint8(color_img))
    color_img = Image.fromarray(imread(mask_path))
    color_img.putpalette(palette)
    color_img.save(vis_path)

def render_dir(mask_dir, vis_dir, palette):
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    mask_list = glob.glob(os.path.join(mask_dir, '*.png'))
    for maskp in mask_list:
        visp = os.path.join(vis_dir, os.path.basename(maskp))
        render_path(maskp, visp, palette)


if __name__ == '__main__':
    from collections import OrderedDict
    COLOR_MAP = OrderedDict(
        ignore = (0,0,0),
        Background=(255, 255, 255),
        Building=(255, 0, 0),
        Road=(255, 255, 0),
        Water=(0, 0, 255),
        Barren=(159, 129, 183),
        Forest=(0, 255, 0),
        Agricultural=(255, 195, 128),
    )

    palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()
    mask_dir = r'D:\Doing\Going\Projects\data\LoveDA\Test\Rural\masks_png'
    vis_dir = r'D:\Doing\Going\Projects\data\LoveDA\Test\Rural\masks_png_color'
    render_dir(mask_dir, vis_dir, palette)