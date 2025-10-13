import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.KAD_Disformer import KAD_Disformer
from utils.dataset import KAD_DisformerTrainSet, KAD_DisformerTestSet

model = KAD_Disformer(
    heads=4,
    d_model=64,
    patch_size=20,
    dropout=0,
    forward_expansion=1,
    encoder_num_layers=1,
    decoder_num_layers=2,
)

# Generate synthetic data: sum of sine waves
t = np.linspace(0, 400, 2000)
raw_series = torch.tensor(np.sin(t) + np.sin(t * 3) + np.sin(t * 5), dtype=torch.float32)

# --- Training ---
train_dataset = KAD_DisformerTrainSet(
    raw_series,
    win_len=20,
    seq_len=100,
    seq_stride=10,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    drop_last=False,
    shuffle=True,
)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("Starting pre-training...")
model.train()
for epoch in range(10):
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/10")
    for i, (seq_context, seq_history, denoised_seq) in enumerate(progress_bar):
        optimizer.zero_grad()
        output = model(seq_context, seq_history)
        loss = criterion(output, denoised_seq)
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(loss=loss.item())

print("Pre-training finished.")

# --- Testing with anomalies ---
print("\nStarting testing with anomalies...")
model.eval()

# Create anomalous data
anomalous_series = raw_series.clone()
anomalous_series[500:510] += 5  # Inject a spike anomaly

test_dataset = KAD_DisformerTestSet(
    anomalous_series,
    win_len=20,
    seq_len=100,
    seq_stride=1,  # Use stride 1 for dense anomaly scores
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    drop_last=False,
    shuffle=False,
)

anomaly_scores = []
all_outputs = []
with torch.no_grad():
    for seq_context, seq_history in tqdm(test_loader, desc="Testing"):
        output = model(seq_context, seq_history)
        # Calculate MSE for each sample in the batch
        mse = torch.mean((output - seq_context) ** 2, dim=[1, 2])
        anomaly_scores.extend(mse.cpu().numpy())
        all_outputs.append(output.cpu())

all_outputs = torch.cat(all_outputs, dim=0)

# Reconstruct the full time series from the output patches by averaging overlaps
reconstructed_full = np.zeros(len(anomalous_series))
counts = np.zeros(len(anomalous_series))

win_len = test_dataset.win_len
seq_len = test_dataset.seq_len
all_outputs_np = all_outputs.numpy()

for i in range(len(all_outputs_np)):  # i is sample index
    for j in range(seq_len):  # j is patch index in sequence
        patch_start_idx = i + j
        patch_end_idx = patch_start_idx + win_len
        if patch_end_idx <= len(reconstructed_full):
            reconstructed_full[patch_start_idx:patch_end_idx] += all_outputs_np[i, j, :]
            counts[patch_start_idx:patch_end_idx] += 1

# Calculate the average for overlapping predictions
reconstructed_avg = np.full(len(anomalous_series), np.nan)
non_zero_indices = counts > 0
reconstructed_avg[non_zero_indices] = reconstructed_full[non_zero_indices] / counts[non_zero_indices]

# --- Visualization ---
print("Generating plot...")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Time Series Reconstruction", "Anomaly Scores"),
                    vertical_spacing=0.15)

# Plot 1: Original and Reconstructed Series
fig.add_trace(
    go.Scatter(x=np.arange(len(anomalous_series)), y=anomalous_series, name="Original Series (with anomaly)",
               line=dict(color='blue')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=np.arange(len(reconstructed_avg)), y=reconstructed_avg, name="Reconstructed Series",
               line=dict(color='red', dash='dash')),
    row=1, col=1
)

# Plot 2: Anomaly Scores
# The score at index `i` corresponds to the window starting at `i`.
score_indices = np.arange(len(reconstructed_avg) - len(anomaly_scores), len(reconstructed_avg))
fig.add_trace(
    go.Scatter(x=score_indices, y=anomaly_scores, name="Anomaly Score (MSE)", line=dict(color='orange')),
    row=2, col=1
)

# Add a shape to highlight the anomaly region
fig.add_vrect(x0=500, x1=510,
              annotation_text="Injected Anomaly", annotation_position="top left",
              fillcolor="red", opacity=0.2, line_width=0, row=1, col=1)
fig.add_vrect(x0=500, x1=510,
              fillcolor="red", opacity=0.2, line_width=0, row=2, col=1)

fig.update_layout(
    title_text="Anomaly Detection Visualization",
    legend_traceorder="reversed",
    height=600,
)
fig.update_yaxes(title_text="Value", row=1, col=1)
fig.update_yaxes(title_text="Score", row=2, col=1)
fig.update_xaxes(title_text="Time Step", row=2, col=1)

fig.show()

max_score = np.max(anomaly_scores)
max_index = np.argmax(anomaly_scores)

print(f"\nMax anomaly score: {max_score:.4f} at sample index: {max_index}")
print("This should correspond to the location of the injected anomaly.")
