- **Download**: [clip](https://huggingface.co/openai/clip-vit-large-patch14) and [michelangelo](https://huggingface.co/Maikou/Michelangelo/tree/main/checkpoints/aligned_shape_latents) model weights
- **Data**: `processed_data.zip`

### Installation

1. Install basic dependencies:
```bash
conda env create -f environment.yml
conda activate env4paper
```

2. Install submodules:
```bash
mkdir submodules && cd submodules
git clone https://github.com/hbb1/diff-surfel-rasterization
git clone https://github.com/graphdeco-inria/simple-knn
pip install ./diff-surfel-rasterization
pip install ./simple-knn
cd ..
```

### Train

```bash
python train.py -s /path/to/dataset -m /path/to/output --ref_ply_path /path/to/anchor --gt_view_names [ground truth images]
```

### Render

```bash
python render.py -s /path/to/dataset -m /path/to/output --gt_view_names [ground truth images]
```
