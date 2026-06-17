#!/usr/bin/env python
"""
prepare_fewshot.py
==================
Run from: mix_gpt2/ directory
    python prepare_fewshot.py

Generates few-shot subsets & YAML configs for:
  - ETTh evening peak   (held-out target)
  - Energy morning peak  (held-out target)

Few-shot sizes: K = 100, 300, 500 rows
"""

import os
import pandas as pd

DATA_DIR = './Data/datasets/'
CONFIG_DIR = './Config/'
FEWSHOT_KS = [100, 300, 500]

YAML_TEMPLATE = """\
model:
  target: Models.interpretable_diffusion.gaussian_diffusion.Diffusion_TS
  params:
    seq_length: 24
    feature_size: 1
    n_layer_enc: 3
    n_layer_dec: 2
    d_model: 64  # 4 X 16
    timesteps: 500
    sampling_timesteps: 500
    loss_type: 'l1'
    beta_schedule: 'cosine'
    n_heads: 4
    mlp_hidden_times: 4
    attn_pd: 0.0
    resid_pd: 0.0
    kernel_size: 1
    padding_size: 0

solver:
  base_lr: 1.0e-5
  max_epochs: 18000
  results_folder: ./Checkpoints_{ckpt}
  gradient_accumulate_every: 2
  save_cycle: 1800
  ema:
    decay: 0.995
    update_interval: 10

  scheduler:
    target: engine.lr_sch.ReduceLROnPlateauWithWarmup
    params:
      factor: 0.5
      patience: 4000
      min_lr: 1.0e-5
      threshold: 1.0e-1
      threshold_mode: rel
      warmup_lr: 8.0e-4
      warmup: 500
      verbose: False

dataloader:
  train_dataset:
    target: Utils.Data_utils.real_datasets.CustomDataset
    params:
      name: {ds_name}
      proportion: 1.0
      data_root: ./Data/datasets/{csv_file}
      window: 24
      save2npy: True
      neg_one_to_one: True
      seed: 123
      period: train

  test_dataset:
    target: Utils.Data_utils.real_datasets.CustomDataset
    params:
      name: {ds_name}
      proportion: 0.9
      data_root: ./Data/datasets/{csv_file}
      window: 24
      save2npy: True
      neg_one_to_one: True
      seed: 123
      period: test
      style: separate
      distribution: geometric
    coefficient: 1.0e-2
    step_size: 5.0e-2
    sampling_steps: 200

  batch_size: 32
  sample_size: 256
  shuffle: True
"""

# Targets: (source_csv, ds_name_for_read_data, config_base_name)
TARGETS = [
    ('evening_peak_etth.csv',    'etth',   'evening_peak_etth'),
    ('morning_peak_energy.csv',  'energy', 'morning_peak_energy'),
]

print('=' * 60)
print('Generating few-shot data & configs')
print('=' * 60)

for src_csv, ds_name, base_name in TARGETS:
    src_path = os.path.join(DATA_DIR, src_csv)
    df = pd.read_csv(src_path)
    total = len(df)
    print(f'\n  [{base_name}] total rows = {total}')

    for k in FEWSHOT_KS:
        if k >= total:
            print(f'    K={k}: skip (>= total rows)')
            continue

        # Take the first K rows (contiguous block for valid windowing)
        subset = df.iloc[:k]
        out_csv = f'fewshot{k}_{base_name}.csv'
        subset.to_csv(os.path.join(DATA_DIR, out_csv), index=False)

        # Write config
        cfg_name = f'fewshot{k}_{base_name}'
        content = YAML_TEMPLATE.format(
            ckpt=cfg_name,
            ds_name=ds_name,
            csv_file=out_csv,
        )
        with open(os.path.join(CONFIG_DIR, f'{cfg_name}.yaml'), 'w') as f:
            f.write(content)

        n_windows = max(k - 24 + 1, 0)
        pct = k / total * 100
        print(f'    K={k}: {out_csv} ({n_windows} windows, {pct:.1f}%)  config -> {cfg_name}.yaml')

print('\n' + '=' * 60)
print('Add these to ett_ins in main.py:')
print('=' * 60)
for _, _, base_name in TARGETS:
    for k in FEWSHOT_KS:
        cfg_name = f'fewshot{k}_{base_name}'
        # Reuse same prompt as the full version
        prompt = 'ETTEP' if 'etth' in base_name else 'ENEMP'
        print(f"  '{cfg_name}': '{prompt}',")

print('\nDone!')
