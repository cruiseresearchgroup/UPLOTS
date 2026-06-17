#!/usr/bin/env python
"""
prepare_new_experiments.py
==========================
Run from: mit_gpt2/ directory
    python prepare_new_experiments.py

Generates:
  1. Fixed workday/weekend energy CSVs (strips date column)
  2. High-load CSVs  (primary feature > 75th percentile)
  3. Low-load CSVs   (primary feature < 25th percentile)
  4. Volatile CSVs    (|first-diff| > 75th percentile)
  5. All corresponding YAML config files
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = './Data/datasets/'
CONFIG_DIR = './Config/'

# ────────────────────────────────────────────────────────────
# YAML template (identical structure to morning_peak_*.yaml)
# ────────────────────────────────────────────────────────────
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
  results_folder: ./Checkpoints_{ckpt_name}
  gradient_accumulate_every: 2
  save_cycle: 1800  # max_epochs // 10
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
      proportion: 1.0  # Set to rate < 1 if training conditional generation
      data_root: ./Data/datasets/{csv_file}
      window: 24  # seq_length
      save2npy: True
      neg_one_to_one: True
      seed: 123
      period: train

  test_dataset:
    target: Utils.Data_utils.real_datasets.CustomDataset
    params:
      name: {ds_name}
      proportion: 0.9  # rate
      data_root: ./Data/datasets/{csv_file}
      window: 24  # seq_length
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


def write_config(config_name, ds_name, csv_file):
    """Write a YAML config file."""
    ckpt_name = config_name.replace('weekend_', 'wke_').replace('workday_', 'wkd_')
    content = YAML_TEMPLATE.format(
        ckpt_name=ckpt_name,
        ds_name=ds_name,
        csv_file=csv_file,
    )
    path = os.path.join(CONFIG_DIR, f'{config_name}.yaml')
    with open(path, 'w') as f:
        f.write(content)
    print(f'    config  -> {config_name}.yaml')


def filter_and_save(df, mask, csv_name, has_date=False, ds_name='energy'):
    """Filter DataFrame by mask, save CSV, create config."""
    subset = df[mask]
    out_path = os.path.join(DATA_DIR, csv_name)
    subset.to_csv(out_path, index=False)
    config_name = csv_name.replace('.csv', '')
    write_config(config_name, ds_name, csv_name)
    return mask.sum()


# ================================================================
print('=' * 60)
print('STEP 1 : Fix workday/weekend Energy CSVs (strip date column)')
print('=' * 60)
for fname in ['workdays_energy.csv', 'non_workdays_energy.csv']:
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        print(f'  [SKIP] {fname} not found'); continue
    df = pd.read_csv(fpath)
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
        df.to_csv(fpath, index=False)
        print(f'  [FIX]  {fname}: stripped date -> {len(df)} rows x {len(df.columns)} cols')
    else:
        print(f'  [OK]   {fname}: {len(df)} rows x {len(df.columns)} cols')

# ================================================================
print('\n' + '=' * 60)
print('STEP 2 : Create workday / weekend configs')
print('=' * 60)
# ETTh  (name=etth -> read_data drops date column)
write_config('workday_etth',    'etth',   'workdays_etth.csv')
write_config('weekend_etth',    'etth',   'non_workdays_etth.csv')
# Energy (name=energy -> no special handling, CSV has no date after fix)
write_config('workday_energy',  'energy', 'workdays_energy.csv')
write_config('weekend_energy',  'energy', 'non_workdays_energy.csv')
# PEMS04 / PEMS08 (follow existing morning_peak pattern)
write_config('workday_pems04',  'pems04', 'workdays_pems04.csv')
write_config('weekend_pems04',  'pems04', 'non_workdays_pems04.csv')
write_config('workday_pems08',  'pems08', 'workdays_pems08.csv')
write_config('weekend_pems08',  'pems08', 'non_workdays_pems08.csv')

# ================================================================
print('\n' + '=' * 60)
print('STEP 3 : Generate high-load / low-load CSVs')
print('=' * 60)

# ---- ETTh ----
print('  [ETTh]')
etth = pd.read_csv(os.path.join(DATA_DIR, 'ETTh.csv'))
ot = etth['OT']
q75, q25 = ot.quantile(0.75), ot.quantile(0.25)
n_hi = filter_and_save(etth, ot >= q75, 'high_load_etth.csv', ds_name='etth')
n_lo = filter_and_save(etth, ot <= q25, 'low_load_etth.csv',  ds_name='etth')
print(f'    high_load={n_hi} rows  low_load={n_lo} rows')

# ---- Energy ----
print('  [Energy]')
energy = pd.read_csv(os.path.join(DATA_DIR, 'energy_data.csv'))
appl = energy.iloc[:, 0]          # Appliances
q75, q25 = appl.quantile(0.75), appl.quantile(0.25)
n_hi = filter_and_save(energy, appl >= q75, 'high_load_energy.csv', ds_name='energy')
n_lo = filter_and_save(energy, appl <= q25, 'low_load_energy.csv',  ds_name='energy')
print(f'    high_load={n_hi} rows  low_load={n_lo} rows')

# ---- PEMS04 ----
print('  [PEMS04]')
pems04_raw = np.load(os.path.join(DATA_DIR, 'PEMS04.npz'))['data'][:, :, 0]
pems04 = pd.DataFrame(pems04_raw)
pm = pems04.mean(axis=1)          # mean traffic flow across all sensors
q75, q25 = pm.quantile(0.75), pm.quantile(0.25)
n_hi = filter_and_save(pems04, pm >= q75, 'high_load_pems04.csv', ds_name='pems04')
n_lo = filter_and_save(pems04, pm <= q25, 'low_load_pems04.csv',  ds_name='pems04')
print(f'    high_load={n_hi} rows  low_load={n_lo} rows')

# ---- PEMS08 ----
print('  [PEMS08]')
pems08_raw = np.load(os.path.join(DATA_DIR, 'PEMS08.npz'))['data'][:, :, 0]
pems08 = pd.DataFrame(pems08_raw)
pm = pems08.mean(axis=1)
q75, q25 = pm.quantile(0.75), pm.quantile(0.25)
n_hi = filter_and_save(pems08, pm >= q75, 'high_load_pems08.csv', ds_name='pems08')
n_lo = filter_and_save(pems08, pm <= q25, 'low_load_pems08.csv',  ds_name='pems08')
print(f'    high_load={n_hi} rows  low_load={n_lo} rows')

# ================================================================
print('\n' + '=' * 60)
print('STEP 4 : Generate volatile (rapid-change) CSVs')
print('=' * 60)

# ---- ETTh ----
print('  [ETTh]')
diff_abs = etth['OT'].diff().abs()
dq75 = diff_abs.quantile(0.75)
n = filter_and_save(etth, diff_abs >= dq75, 'volatile_etth.csv', ds_name='etth')
print(f'    volatile={n} rows')

# ---- Energy ----
print('  [Energy]')
diff_abs = energy.iloc[:, 0].diff().abs()
dq75 = diff_abs.quantile(0.75)
n = filter_and_save(energy, diff_abs >= dq75, 'volatile_energy.csv', ds_name='energy')
print(f'    volatile={n} rows')

# ---- PEMS04 ----
print('  [PEMS04]')
diff_abs = pems04.mean(axis=1).diff().abs()
dq75 = diff_abs.quantile(0.75)
n = filter_and_save(pems04, diff_abs >= dq75, 'volatile_pems04.csv', ds_name='pems04')
print(f'    volatile={n} rows')

# ---- PEMS08 ----
print('  [PEMS08]')
diff_abs = pems08.mean(axis=1).diff().abs()
dq75 = diff_abs.quantile(0.75)
n = filter_and_save(pems08, diff_abs >= dq75, 'volatile_pems08.csv', ds_name='pems08')
print(f'    volatile={n} rows')

# ================================================================
print('\n' + '=' * 60)
print('SUMMARY : New ett_ins entries for main.py')
print('=' * 60)
new_entries = [
    # workday / weekend
    ("'workday_etth': 'ETTWKD'",    "'weekend_etth': 'ETTWKE'"),
    ("'workday_energy': 'ENEWKD'",  "'weekend_energy': 'ENEWKE'"),
    ("'workday_pems04': 'P04WKD'",  "'weekend_pems04': 'P04WKE'"),
    ("'workday_pems08': 'P08WKD'",  "'weekend_pems08': 'P08WKE'"),
    # high-load / low-load
    ("'high_load_etth': 'ETTHI'",   "'low_load_etth': 'ETTLO'"),
    ("'high_load_energy': 'ENEHI'", "'low_load_energy': 'ENELO'"),
    ("'high_load_pems04': 'P04HI'", "'low_load_pems04': 'P04LO'"),
    ("'high_load_pems08': 'P08HI'", "'low_load_pems08': 'P08LO'"),
    # volatile
    ("'volatile_etth': 'ETTVOL'",   "'volatile_energy': 'ENEVOL'"),
    ("'volatile_pems04': 'P04VOL'", "'volatile_pems08': 'P08VOL'"),
]
for a, b in new_entries:
    print(f'  {a}, {b},')

print('\n' + '=' * 60)
print('All done!  New CSVs in Data/datasets/  |  New configs in Config/')
print('=' * 60)
