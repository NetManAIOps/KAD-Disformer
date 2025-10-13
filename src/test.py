import argparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.KAD_Disformer import KAD_Disformer
from utils.dataset import KAD_DisformerTestSet
from utils.evaluate import f1_score_with_point_adjust, f1_score_point

def main(args):
    # Load model
    model = KAD_Disformer(
        heads=args.heads,
        d_model=args.d_model,
        patch_size=args.patch_size,
        dropout=args.dropout,
        forward_expansion=args.forward_expansion,
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
    )
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)
    model.eval()

    # Load data
    test_df = pd.read_csv(args.data_path)
    raw_series = torch.tensor(test_df["value"].to_numpy(), dtype=torch.float32)

    test_dataset = KAD_DisformerTestSet(
        raw_series,
        win_len=args.win_len,
        seq_len=args.seq_len,
        seq_stride=args.seq_stride,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=False,
    )

    print("Starting testing...")
    all_outputs = []
    with torch.no_grad():
        for seq_context, seq_history in tqdm(test_loader, desc="Testing"):
            seq_context = seq_context.to(args.device)
            seq_history = seq_history.to(args.device)
            output = model(seq_context, seq_history)
            # Calculate MSE for each sample in the batch
            mse = torch.mean((output - seq_context) ** 2, dim=[1, 2])
            all_outputs.append(output.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)

    # Reconstruct the full time series from the output patches by averaging overlaps
    reconstructed_full = np.zeros(len(raw_series))
    counts = np.zeros(len(raw_series))

    win_len = test_dataset.win_len
    seq_len = test_dataset.seq_len
    all_outputs_np = all_outputs.numpy()

    for i in range(len(all_outputs_np)):  # i is sample index
        for j in range(seq_len):  # j is patch index in sequence
            patch_start_idx = i * args.seq_stride + j
            patch_end_idx = patch_start_idx + win_len
            if patch_end_idx <= len(reconstructed_full):
                reconstructed_full[patch_start_idx:patch_end_idx] += all_outputs_np[i, j, :]
                counts[patch_start_idx:patch_end_idx] += 1

    # Calculate the average for overlapping predictions
    reconstructed_avg = np.full(len(raw_series), np.nan)
    non_zero_indices = counts > 0
    reconstructed_avg[non_zero_indices] = reconstructed_full[non_zero_indices] / counts[non_zero_indices]

    anomaly_scores = np.abs(reconstructed_avg - raw_series.cpu().numpy())
    # --- Visualization ---
    print(f"Generating plot and saving to {args.output_plot_path}...")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Time Series Reconstruction", "Anomaly Scores"),
                        vertical_spacing=0.15)

    # Plot 1: Original and Reconstructed Series
    fig.add_trace(
        go.Scatter(x=np.arange(len(raw_series)), y=raw_series, name="Original Series",
                   line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=np.arange(len(reconstructed_avg)), y=reconstructed_avg, name="Reconstructed Series",
                   line=dict(color='red', dash='dash')),
        row=1, col=1
    )

    # Plot 2: Anomaly Scores
    score_indices = np.arange(len(anomaly_scores)) * args.seq_stride
    fig.add_trace(
        go.Scatter(x=score_indices, y=anomaly_scores, name="Anomaly Score (MSE)", line=dict(color='orange')),
        row=2, col=1
    )

    fig.update_layout(
        title_text="Anomaly Detection Visualization",
        legend_traceorder="reversed",
        height=600,
    )
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    fig.update_xaxes(title_text="Time Step", row=2, col=1)

    fig.write_html(args.output_plot_path)

    # --- Evaluation ---
    print("Evaluating model...")
    labels = test_df["label"].to_numpy()
    f1_adjust = f1_score_with_point_adjust(labels, anomaly_scores, delay=args.delay)
    f1 = f1_score_point(labels, anomaly_scores)

    print(f"Precision, Recall, F1 Score with Point Adjustment (delay={args.delay}): {f1_adjust['p']:.4f}, {f1_adjust['r']:.4f}, {f1_adjust['f']:.4f}")
    print(f"Precision, Recall, F1 Score (no point adjustment): {f1['p']:.4f}, {f1['r']:.4f}, {f1['f']:.4f}")

    print("Done.")


def parse_args():
    parser = argparse.ArgumentParser(description='KAD-Disformer Testing')
    # Model parameters
    parser.add_argument('--heads', type=int, default=4, help='Number of heads in Multi-Head Attention')
    parser.add_argument('--d_model', type=int, default=64, help='Dimension of model')
    parser.add_argument('--patch_size', type=int, default=20, help='Patch size')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--forward_expansion', type=int, default=1,
                        help='Forward expansion factor in FeedForward layer')
    parser.add_argument('--encoder_num_layers', type=int, default=1, help='Number of encoder layers')
    parser.add_argument('--decoder_num_layers', type=int, default=2, help='Number of decoder layers')
    parser.add_argument('--device', type=str, default='cpu', help='Device to train on (e.g., "cpu", "cuda")')
    parser.add_argument('--model_path', type=str, default='checkpoints/model_final.pth', help='Path to trained model')

    # Data parameters
    parser.add_argument('--data_path', type=str, default='data/test.csv', help='Path to test data')
    parser.add_argument('--win_len', type=int, default=20, help='Window length for dataset')
    parser.add_argument('--seq_len', type=int, default=100, help='Sequence length for dataset')
    parser.add_argument('--seq_stride', type=int, default=1, help='Sequence stride for dataset')

    # Testing parameters
    parser.add_argument('--output_plot_path', type=str, default='test_plot.html', help='Path to save output plot')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for testing')
    parser.add_argument('--delay', type=int, default=10, help='Delay for F1 score with point adjustment')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
