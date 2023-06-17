from ever.core.checkpoint import load_model_state_dict_from_ckpt
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch
sns.set()
save_dir = r'C:\Users\zouda\Desktop\Conference\NIPS\figs\appendix\transnorm'


import glob
pngp_list = glob.glob(os.path.join(save_dir, '*.png'))
for pngp in pngp_list:
    new_pngp = pngp.replace('.', '_').replace('_png', '.png')
    print(new_pngp)
    os.rename(pngp, new_pngp)