#!/usr/bin/env python
"""
evaluate.py — Compute C-FID, Discriminative, Predictive scores
===============================================================
Usage:
    python evaluate.py --ori <ori.npy> --fake <fake.npy> --tag "zeroshot_ettep"
    python evaluate.py --ori <ori.npy> --fake <fake.npy> --tag "x" --metrics discri

Can be called from any project dir (mix_gpt2 or mix_diffts_glab)
as long as sys.path includes the project root.
"""
import os, sys, json, argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori',  required=True, help='Path to ori norm_truth npy')
    parser.add_argument('--fake', required=True, help='Path to fake npy')
    parser.add_argument('--tag',  default='', help='Label for this evaluation')
    parser.add_argument('--iters', type=int, default=5, help='Number of iterations')
    parser.add_argument('--out', default=None, help='Output JSON path (optional)')
    parser.add_argument('--metrics', nargs='+', default=['cfid', 'discri', 'predic'],
                        choices=['cfid', 'discri', 'predic'], help='Metrics to compute')
    args = parser.parse_args()

    ori_data = np.load(args.ori)
    fake_data = np.load(args.fake)
    print(f'[{args.tag}] ori={ori_data.shape}  fake={fake_data.shape}')

    # Flatten multivariate -> univariate (same as notebooks)
    b, t, n = ori_data.shape
    ori_flat = ori_data.transpose(2, 0, 1).reshape(b * n, t, 1)
    fake_flat = fake_data[:ori_flat.shape[0]]
    print(f'  After flatten: ori={ori_flat.shape}  fake={fake_flat.shape}')

    results = {'tag': args.tag, 'ori_shape': list(ori_data.shape), 'fake_shape': list(fake_data.shape)}

    # ---- C-FID (PyTorch) ----
    if 'cfid' not in args.metrics:
        print('  [SKIP] C-FID (not requested)')
    else:
        try:
            from Utils.context_fid import Context_FID
            from Utils.metric_utils import display_scores as ds
            scores = []
            for i in range(args.iters):
                s = Context_FID(ori_flat, fake_flat)
                scores.append(s)
                print(f'  C-FID iter {i}: {s:.4f}')
            mean_s, std_s = np.mean(scores), np.std(scores)
            print(f'  C-FID = {mean_s:.4f} +/- {std_s:.4f}')
            results['cfid_mean'] = float(mean_s)
            results['cfid_std'] = float(std_s)
        except Exception as e:
            print(f'  [SKIP] C-FID failed: {e}')

    # ---- Discriminative Score (TensorFlow) ----
    if 'discri' not in args.metrics:
        print('  [SKIP] Discriminative (not requested)')
    else:
        try:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            from Utils.discriminative_metric import discriminative_score_metrics
            scores = []
            for i in range(args.iters):
                s, fa, ra = discriminative_score_metrics(ori_flat, fake_flat)
                scores.append(s)
                print(f'  Discri iter {i}: {s:.4f}  (fake_acc={fa:.3f}, real_acc={ra:.3f})')
            mean_s, std_s = np.mean(scores), np.std(scores)
            print(f'  Discriminative = {mean_s:.4f} +/- {std_s:.4f}')
            results['discri_mean'] = float(mean_s)
            results['discri_std'] = float(std_s)
        except Exception as e:
            print(f'  [SKIP] Discriminative failed: {e}')

    # ---- Predictive Score (TensorFlow) ----
    if 'predic' not in args.metrics:
        print('  [SKIP] Predictive (not requested)')
    else:
        try:
            from Utils.predictive_metric import predictive_score_metrics
            scores = []
            for i in range(args.iters):
                s = predictive_score_metrics(ori_flat, fake_flat)
                scores.append(s)
                print(f'  Predic iter {i}: {s:.4f}')
            mean_s, std_s = np.mean(scores), np.std(scores)
            print(f'  Predictive = {mean_s:.4f} +/- {std_s:.4f}')
            results['predic_mean'] = float(mean_s)
            results['predic_std'] = float(std_s)
        except Exception as e:
            print(f'  [SKIP] Predictive failed: {e}')

    # ---- Save results ----
    out_path = args.out or f'eval_{args.tag}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  Results saved to {out_path}')
    print('=' * 60)

if __name__ == '__main__':
    main()
