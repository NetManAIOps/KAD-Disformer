FROM python:3.12-slim

# Install uv, a fast Python package installer and resolver
RUN pip install uv

# Set the working directory
WORKDIR /app

# Copy the entire project source code
COPY . .

# Install dependencies using uv
RUN uv sync

# Create a script to run the full reproduction pipeline
# This script executes pre-training, fine-tuning, and testing in sequence.
RUN echo '#!/bin/sh' > run.sh && \
    echo 'set -e' >> run.sh && \
    echo '' >> run.sh && \
    echo 'echo "--- Running Pre-training ---"' >> run.sh && \
    echo 'uv run src/train.py --data_path data/train.csv --checkpoint_dir checkpoints' >> run.sh && \
    echo '' >> run.sh && \
    echo 'echo "--- Pre-training complete, starting Fine-tuning ---"' >> run.sh && \
    echo 'uv run src/finetune.py --model_path checkpoints/model_final.pth --finetune_data_path data/finetune_train.csv --test_data_path data/finetune_test.csv' >> run.sh && \
    echo '' >> run.sh && \
    echo 'echo "--- All steps complete ---"' >> run.sh && \
    chmod +x run.sh

# Set the default command to execute the run script
CMD ["./run.sh"]