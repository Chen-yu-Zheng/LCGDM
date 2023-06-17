# Domain Adaptive Land-Cover Classification via Local Consistency and Global Diversity

This is the official implementation for the TGRS 2023 paper [Domain Adaptive Land-Cover Classification via Local Consistency and Global Diversity](https://ieeexplore.ieee.org/document/10094018).

## Dependencies

* Pytorch
* albumentations
* sklearn
* skimage
* [ever](https://github.com/Z-Zheng/ever)
* [LoveDA dataset](https://github.com/Junjue-Wang/LoveDA)

## Training with different UDA methods (baseline, +Ent, +LCGDM)

For example, to reproduce the results of CBST in the paper, one can run

```bash
bash scripts/cbst/train_cbst.sh #baseline
bash scripts/cbst/train_cbst_Ent.sh # +Ent
bash scripts/cbst/train_cbst_SL.sh # +LCGDM
```

## Inference on the test set

Submit your test results on  [**LoveDA Unsupervised Domain Adaptation Challenge**](https://codalab.lisn.upsaclay.fr/competitions/424) and obtain the final score.

```bash
python predict.py # you should set the arguments (e.g. model path) 
```

## T-SNE visualization

```bash
python tsne.py # CBST and CLAN are supported, you should set the model path and the image path
```

## Hyper-parameters Configuration

Detailed hyperparameters config can be found in folder "configs/LoveDA".

## Citation
If you use our code in your research, please cite our TGRS 2023 paper.
```text
@article{DBLP:journals/tgrs/MaZWZ23,
  author       = {Ailong Ma and
                  Chenyu Zheng and
                  Junjue Wang and
                  Yanfei Zhong},
  title        = {Domain Adaptive Land-Cover Classification via Local Consistency and
                  Global Diversity},
  journal      = {{IEEE} Trans. Geosci. Remote. Sens.},
  volume       = {61},
  pages        = {1--17},
  year         = {2023}
}
```

## Acknowledgments

The code is developed based on the following repositories. We appreciate their nice implementations.

| Method |              Repository               |
| :----: | :-----------------------------------: |
| LoveDA | https://github.com/Junjue-Wang/LoveDA |
| LoveCS | https://github.com/Junjue-Wang/LoveCS |
|  DCA   |    https://github.com/Luffy03/DCA     |

