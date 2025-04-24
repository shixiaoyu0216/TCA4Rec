Code for our Journal of Big Data 2025 Paper[**“TCA4Rec: Contrastive Learning with Popularity-aware Asymmetric Augmentation for Robust Sequential Recommendation”**]


<img src="figs/Framework.png" alt="Framework" style="zoom:100%;" />

## Installation

```bash
torch==1.1.0
numpy==1.19.1
scipy==1.5.2
tqdm==4.48.2
```

## Examples to run the code

- #### Sports

```bash
python run_mminforec.py --lr=0.001 --weight_decay=1e-3 --pred_step=1 --tau=0.6 --data_name=Sports_and_Outdoors --num_hidden_layers=1 --num_attention_heads=1 --dc_s=1 --dc=1 --num_hidden_layers_gru=1 --mil=4 --epoch=200 --loss_fuse_dropout_prob=0.5 --mem=64
```

## Credit

This repo is based on [MMInfoRec]
