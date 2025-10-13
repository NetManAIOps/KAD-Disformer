import argparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.KAD_Disformer import KAD_Disformer
from utils.config import setup_seed
from utils.dataset import KAD_DisformerTrainSet, KAD_DisformerTestSet
from utils.evaluate import f1_score_with_point_adjust, f1_score_point


def evaluate(model, args, data_path, plot_path=None):
    """Evaluates the model on the given data."""
    model.eval()

    # Load data
    test_df = pd.read_csv(data_path)
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

    print(f"Starting evaluation on {data_path}...")
    all_outputs = []
    with torch.no_grad():
        for seq_context, seq_history in tqdm(test_loader, desc="Evaluating"):
            seq_context = seq_context.to(args.device)
            seq_history = seq_history.to(args.device)
            output = model(seq_context, seq_history)
            all_outputs.append(output.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)

    # Reconstruct
    reconstructed_full = np.zeros(len(raw_series))
    counts = np.zeros(len(raw_series))
    win_len = test_dataset.win_len
    seq_len = test_dataset.seq_len
    all_outputs_np = all_outputs.numpy()

    for i in range(len(all_outputs_np)):
        for j in range(seq_len):
            patch_start_idx = i * args.seq_stride + j
            patch_end_idx = patch_start_idx + win_len
            if patch_end_idx <= len(reconstructed_full):
                reconstructed_full[patch_start_idx:patch_end_idx] += all_outputs_np[i, j, :]
                counts[patch_start_idx:patch_end_idx] += 1

    reconstructed_avg = np.full(len(raw_series), np.nan)
    non_zero_indices = counts > 0
    reconstructed_avg[non_zero_indices] = reconstructed_full[non_zero_indices] / counts[non_zero_indices]

    anomaly_scores = np.abs(reconstructed_avg - raw_series.cpu().numpy())

    if 'label' in test_df.columns:
        labels = test_df["label"].to_numpy()
        f1_adjust = f1_score_with_point_adjust(labels, anomaly_scores, delay=args.delay)
        f1 = f1_score_point(labels, anomaly_scores)
        print(
            f"Precision, Recall, F1 Score with Point Adjustment (delay={args.delay}): {f1_adjust['p']:.4f}, {f1_adjust['r']:.4f}, {f1_adjust['f']:.4f}")
        print(f"Precision, Recall, F1 Score (no point adjustment): {f1['p']:.4f}, {f1['r']:.4f}, {f1['f']:.4f}")
    else:
        f1_adjust = f1 = {'p': 0, 'r': 0, 'f': 0}
        print("No labels found in test data. Skipping F1 score calculation.")

    if plot_path:
        print(f"Generating plot and saving to {plot_path}...")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Time Series Reconstruction", "Anomaly Scores"),
                            vertical_spacing=0.15)
        fig.add_trace(
            go.Scatter(x=np.arange(len(raw_series)), y=raw_series, name="Original Series", line=dict(color='blue')),
            row=1, col=1)
        fig.add_trace(go.Scatter(x=np.arange(len(reconstructed_avg)), y=reconstructed_avg, name="Reconstructed Series",
                                 line=dict(color='red', dash='dash')), row=1, col=1)
        score_indices = np.arange(len(anomaly_scores))
        fig.add_trace(
            go.Scatter(x=score_indices, y=anomaly_scores, name="Anomaly Score", line=dict(color='orange')),
            row=2, col=1)
        fig.update_layout(title_text="Anomaly Detection Visualization", height=600)
        fig.write_html(plot_path)

    return f1_adjust, f1


def main(args):
    setup_seed(args.seed)

    # Load pre-trained model
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

    print("--- Performance Before Finetuning ---")
    evaluate(model, args, args.test_data_path, plot_path="finetune_before_plot.html")

    # Freeze parameters
    for name, param in model.named_parameters():
        if 'adapter' not in name and 'personal' not in name:
            param.requires_grad = False

    # Print trainable parameters
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    # --- Finetuning ---
    train_df = pd.read_csv(args.finetune_data_path)
    raw_series = train_df["value"].to_numpy()

    train_dataset = KAD_DisformerTrainSet(
        raw_series,
        win_len=args.win_len,
        seq_len=args.seq_len,
        seq_stride=args.seq_stride,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=True,
    )

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=0.1)
    criterion = nn.MSELoss()

    print("\nStarting finetuning...")
    model.train()
    for epoch in range(args.epochs):
        progress_bar = tqdm(train_loader, desc=f"Finetune Epoch {epoch + 1}/{args.epochs}")
        for i, (seq_context, seq_history, denoised_seq) in enumerate(progress_bar):
            optimizer.zero_grad()
            seq_context = seq_context.to(args.device)
            seq_history = seq_history.to(args.device)
            denoised_seq = denoised_seq.to(args.device)

            output = model(seq_context, seq_history)
            loss = criterion(output, denoised_seq)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())
        scheduler.step()

    print("Finetuning finished.")

    print("\n--- Performance After Finetuning ---")
    evaluate(model, args, args.test_data_path, plot_path="finetune_after_plot.html")


def parse_args():
    parser = argparse.ArgumentParser(description='KAD-Disformer Finetuning')
    # Model parameters
    parser.add_argument('--heads', type=int, default=4, help='Number of heads in Multi-Head Attention')
    parser.add_argument('--d_model', type=int, default=64, help='Dimension of model')
    parser.add_argument('--patch_size', type=int, default=20, help='Patch size')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--forward_expansion', type=int, default=1, help='Forward expansion factor')
    parser.add_argument('--encoder_num_layers', type=int, default=1, help='Number of encoder layers')
    parser.add_argument('--decoder_num_layers', type=int, default=2, help='Number of decoder layers')
    parser.add_argument('--device', type=str, default='cpu', help='Device to train on')
    parser.add_argument('--model_path', type=str, default='checkpoints/model_final.pth',
                        help='Path to pre-trained model')

    # Data parameters
    parser.add_argument('--finetune_data_path', type=str, default='data/finetune_train.csv',
                        help='Path to finetuning training data')
    parser.add_argument('--test_data_path', type=str, default='data/finetune_test.csv',
                        help='Path to finetuning test data')
    parser.add_argument('--win_len', type=int, default=20, help='Window length')
    parser.add_argument('--seq_len', type=int, default=100, help='Sequence length')
    parser.add_argument('--seq_stride', type=int, default=1, help='Sequence stride')

    # Finetuning parameters
    parser.add_argument('--epochs', type=int, default=5, help='Number of finetuning epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--lr_step_size', type=int, default=3, help='Step size for learning rate scheduler')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed')
    parser.add_argument('--delay', type=int, default=10, help='Delay for F1 score with point adjustment')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
