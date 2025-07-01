import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import socket
import struct
import evaluate
import config
import csv
import shutil
from tqdm import tqdm
from datetime import datetime
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    OPTForSequenceClassification,
    DataCollatorWithPadding,
    set_seed,
)
from torch.utils.data import DataLoader
from utils import load_and_split_dataset

# Set Seed
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
set_seed(config.SEED)

# Output Directory
base_output_dir = "output"
os.makedirs(base_output_dir, exist_ok=True)
existing_runs = [d for d in os.listdir(base_output_dir) if d.startswith("train_")]
next_run_id = max([int(d.split("_")[1]) for d in existing_runs], default=0) + 1
run_dir = os.path.join(base_output_dir, f"train_{next_run_id}")
os.makedirs(run_dir, exist_ok=True)
shutil.copy("config.py", os.path.join(run_dir, "config.py"))

# Logging
log_path = os.path.join(run_dir, "log_zo.csv")
with open(log_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "val_acc", "test_acc", "timestamp"])
    writer.writeheader()

# Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
model = OPTForSequenceClassification.from_pretrained(config.MODEL_PATH).to("cuda")
full_layers = model.model.decoder.layers
split_idx = config.SPLIT_LAYER_IDX
early_layers = full_layers[:split_idx]
embed = model.model.decoder.embed_tokens
params = list(embed.parameters()) + [p for l in early_layers for p in l.parameters()]

# Load Dataset
dataset: DatasetDict = load_and_split_dataset(
    dataset_path=config.DATASET_PATH,
    seed=config.SEED,
    total_size=config.TOTAL_DATASET,
    train_ratio=config.TRAIN_RATIO,
    val_ratio=config.VAL_RATIO,
    test_ratio=config.TEST_RATIO,
)

def tokenize(batch):
    return tokenizer(batch["sentence"], truncation=True, padding='max_length', max_length=128)

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["sentence"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(tokenized_dataset["train"], batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(tokenized_dataset["validation"], batch_size=config.BATCH_SIZE, collate_fn=data_collator)
test_loader = DataLoader(tokenized_dataset["test"], batch_size=config.BATCH_SIZE, collate_fn=data_collator)

# Socket Communication
HOST = "127.0.0.1"
PORT = 9999
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))

def safe_sendall(data):
    try:
        sock.sendall(data)
    except (BrokenPipeError, OSError) as e:
        print(f"❌ Error during sendall: {e}")
        sock.close()
        raise

def send_tensor(tensor):
    try:
        tensor = tensor.detach().cpu().contiguous()
        data = tensor.numpy().astype(np.float32).tobytes()
        safe_sendall(struct.pack('!I', len(data)) + data)
    except Exception as e:
        print(f"❌ Failed to send tensor: {e}")
        raise

def send_label_tensor(labels):
    try:
        labels = labels.detach().cpu().contiguous()
        data = labels.numpy().astype(np.int64).tobytes()
        safe_sendall(struct.pack('!I', len(data)) + data)
    except Exception as e:
        print(f"❌ Failed to send label tensor: {e}")
        raise

def recv_float():
    data = b''
    while len(data) < 8:
        packet = sock.recv(8 - len(data))
        if not packet:
            raise ConnectionError("Connection lost while receiving float")
        data += packet
    return struct.unpack('!d', data)[0]

def recv_pred_batch(size):
    data = b""
    while len(data) < 4 * size:
        packet = sock.recv(4 * size - len(data))
        if not packet:
            raise ConnectionError("Failed to receive predictions")
        data += packet
    return list(struct.unpack(f'!{size}I', data))

# Node1 Forward
def node1_forward(input_ids):
    with torch.no_grad():
        x = embed(input_ids)
        for layer in early_layers:
            x = layer(x)[0]
    return x

# ZO Utilities
def params_to_vector(params):
    return torch.cat([p.view(-1) for p in params])

def vector_to_params(vector, params_template):
    idx = 0
    new_params = []
    for p in params_template:
        numel = p.numel()
        new_p = vector[idx:idx + numel].view_as(p).to(p.device)
        new_params.append(new_p)
        idx += numel
    return new_params

def update_model_params(params, new_params):
    with torch.no_grad():
        for p, new_p in zip(params, new_params):
            p.copy_(new_p)

# ZO Hyperparameters
mu = config.ZO_MU
lr = config.LEARNING_RATE
P = config.ZO_PERTURBATIONS

# Evaluation
metric = evaluate.load("accuracy")

def evaluate_model(loader):
    model.eval()
    preds, labels_all = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to("cuda")
        labels = batch["labels"].to("cuda")
        h = node1_forward(input_ids)

        try:
            safe_sendall(b'I')
            send_tensor(h)
            send_label_tensor(labels)

            pred_batch = recv_pred_batch(len(labels))
            preds.extend(pred_batch)
            labels_all.extend(labels.cpu().tolist())
        except Exception as e:
            print(f"❌ Error during inference: {e}")
            break

    return metric.compute(predictions=preds, references=labels_all)

def check_param_update_effectiveness(params, w_before, step_id=""):
    w_after = torch.cat([p.view(-1) for p in params]).detach()
    delta = w_after - w_before
    num_changed = (delta.abs() > 1e-6).sum().item()
    total_params = delta.numel()
    print(f"\n🔍 Param Update Check {step_id}")
    print(f"Δ mean: {delta.mean().item():.6e} | Δ std: {delta.std().item():.6e} | |Δ|max: {delta.abs().max().item():.6e}")
    print(f"Δ changed: {num_changed}/{total_params} ({(100 * num_changed / total_params):.2f}%)")
    print(f"Δ preview: {delta[:5]}")

# Training
print("Training ZO-SGD")
try:
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}")
        epoch_losses = []

        for step, batch in enumerate(tqdm(train_loader)):
            input_ids = batch["input_ids"].to("cuda")
            labels = batch["labels"].to("cuda")

            w = params_to_vector(params)
            grad = torch.zeros_like(w)
            losses = []

            for i in range(P):
                print(f"Step {step + 1} | Perturbation {i + 1}/{P}")
                u = torch.randn_like(w)
                w_pos = w + mu * u
                w_neg = w - mu * u

                update_model_params(params, vector_to_params(w_pos, params))
                h_pos = node1_forward(input_ids)

                update_model_params(params, vector_to_params(w_neg, params))
                h_neg = node1_forward(input_ids)

                try:
                    safe_sendall(b'Z')
                    send_tensor(h_pos)
                    send_label_tensor(labels)
                    send_tensor(h_neg)
                    send_label_tensor(labels)

                    print("Waiting for loss from Node2...")
                    L = recv_float()
                    print(f"Received loss: {L:.4f}")
                    grad += L * u
                    losses.append(abs(L))
                except Exception as e:
                    print(f"❌ Error during ZO communication: {e}")
                    break

            if len(losses) == 0:
                print("⚠️ Skipping step due to communication failure.")
                continue

            grad /= P
            avg_loss = np.mean(losses)
            epoch_losses.append(avg_loss)

            w_before = params_to_vector(params).clone()
            new_w = w - lr * grad
            update_model_params(params, vector_to_params(new_w, params))

            check_param_update_effectiveness(params, w_before, step_id=f"Epoch {epoch+1}, Step {step+1}")

            if (step + 1) % 10 == 0:
                print(f"Step {step + 1:03} | Loss: {avg_loss:.4f}")

        val_acc = evaluate_model(val_loader)["accuracy"]
        test_acc = evaluate_model(test_loader)["accuracy"]
        epoch_loss = np.mean(epoch_losses)

        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "val_acc", "test_acc", "timestamp"])
            writer.writerow({
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "timestamp": datetime.now().isoformat(timespec="seconds")
            })

except Exception as e:
    print(f"❌ Training aborted: {e}")
finally:
    sock.close()
    print("🔒 Socket closed")

# Save Model
model.save_pretrained(os.path.join(run_dir, "opt-sst2-zo-finetuned"))
tokenizer.save_pretrained(os.path.join(run_dir, "opt-sst2-zo-finetuned"))
print(f"Model saved to {run_dir}/opt-sst2-zo-finetuned")
print(f"Logs saved to {log_path}")