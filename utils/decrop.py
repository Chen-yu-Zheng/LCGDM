import os
import glob
import random
from skimage.io import imread,imsave
import numpy as np


root_dir = r'J:\2021LoveDA\LoveDA_Test'
maskp_list = glob.glob(os.path.join(root_dir, '*png'))
for maskp in maskp_list:
    mask = imread(maskp).astype(np.float) - 1
    mask[mask==-1] = 0
    imsave(os.path.join(r'J:\2021LoveDA\LoveDA_Sub', os.path.basename(maskp)), mask.astype(np.uint8))

