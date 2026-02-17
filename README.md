# MACE Training + Freeze + Active Learning + “Model Merge” (Committee Disagreement)

This repo provides a reproducible workflow to:
1) split a dataset
2) train a MACE model (reproducibly)
3) optionally freeze parts of a model for fine-tuning
4) run a committee of models and compute disagreement (“model merge” idea)
5) select top-K uncertain structures for active learning
6) repeat

> **Environment note (Windows + AMD GPU):**
> - `--device cuda` is NVIDIA-only.
> - On Windows with AMD GPU, use `--device cpu` for MACE/PyTorch unless you have a special GPU stack.
> - All commands below use `--device cpu` for reliability.

---

## Files in this repo

- `split_dataset.py`  
  Split one dataset into `train.xyz` and `valid.xyz`.

- `mace_train.py`  
  Reproducible training wrapper. Writes `manifest.json` and calls `mace_run_train` internally.

- `inference_test.py`  
  Tests if the outputted mace model can make inferences

- `mace_freeze.py`  
  Creates a “freeze-init” checkpoint and freeze plan metadata (used when fine-tuning).

- `model_disagreement.py`  
  “Model merge” = run multiple models (committee) on the same structures and measure disagreement.

- `mace_active_learning.py`  
  Uses committee disagreement to pick top-K structures to label next.

---

## Step 0: Install dependencies

```bash
pip install mace-torch ase torch numpy
```
## Step 1: Split dataset (if not split already)
```bash
python split_dataset.py
  --input data/Liquid_Water.xyz
  --train_out data/train.xyz
  --valid_out data/valid.xyz
  --valid_fraction 0.1
```
This means that 90% of the data will be for training while 10% is for validation

## Step 2: Train your first model!
```bash
python -u mace_train.py 
  --train_file data/train.xyz 
  --valid_file data/valid.xyz 
  --work_dir runs 
  --name water_1k_small 
  --seed 123 
  --device cpu 
  --extra 
    --E0s average 
    --model MACE 
    --num_interactions 2 
    --num_channels 64 
    --max_L 0 
    --correlation 3 
    --r_max 6.0 
    --forces_weight 1000 
    --energy_weight 10 
    --energy_key TotEnergy 
    --forces_key force 
    --batch_size 2 
    --valid_batch_size 4 
    --max_num_epochs 800 
    --start_swa 400 
    --scheduler_patience 15 
    --patience 30 
    --eval_interval 4 
    --ema 
    --swa 
    --error_table PerAtomMAE 
    --default_dtype float64 
    --restart_latest 
    --save_cpu
```
I know it looks like a lot, but a lot of these specifications are just very picky alterations to make sure everything stays consistent.

This will create a directory like this:
```bash
runs/water_1k_small/
  manifest.json
  checkpoints/
  logs...
```
## Step 2.5: Run a quick test to see if it worked

Step 4: (Optional) Freeze parts of a model (fine-tune)