- **Environment**: `environment.yml`
- **Download**: [clip](https://huggingface.co/openai/clip-vit-large-patch14) and [michelangelo](https://huggingface.co/Maikou/Michelangelo/tree/main/checkpoints/aligned_shape_latents) model weights
- **Data**: Preprocessed DTU data is in `./processed_data/`

### train

```bash
python train.py -s /path/to/dataset -m /path/to/output --ref_ply_path /path/to/anchor --gt_view_names [ground truth images]
```

### render

```bash
python train.py -s /path/to/dataset -m /path/to/output --gt_view_names [ground truth images]
```
