import socket
import struct
import numpy as np
import torch
import torch.nn.functional as F
from transformers import OPTForSequenceClassification
import config

# Setup
HOST = "0.0.0.0"
PORT = 9999
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and extract later layers
model = OPTForSequenceClassification.from_pretrained(config.MODEL_PATH).to(device)
model.eval()

split_idx = config.SPLIT_LAYER_IDX
later_layers = model.model.decoder.layers[split_idx:]
final_norm = model.model.decoder.final_layer_norm
classifier_head = model.score

params = list(p for l in later_layers for p in l.parameters()) + \
         list(final_norm.parameters()) + list(classifier_head.parameters())

hidden_size = model.config.hidden_size
seq_len = 128

# Socket utilities
def recv_tensor(conn, dtype, shape=None):
    length_bytes = conn.recv(4)
    if not length_bytes:
        raise ConnectionError("Failed to receive length header")
    length = struct.unpack('!I', length_bytes)[0]
    data = b''
    while len(data) < length:
        packet = conn.recv(length - len(data))
        if not packet:
            raise ConnectionError("Socket closed while receiving tensor")
        data += packet
    array = np.frombuffer(data, dtype=dtype)
    tensor = torch.from_numpy(array).to(device)
    if shape:
        tensor = tensor.contiguous().view(*shape)
    return tensor

def recv_hidden(conn, batch_size=None):
    array = recv_tensor(conn, np.float32)
    if batch_size is None:
        total = array.numel()
        batch_size = total // (seq_len * hidden_size)
    return array.contiguous().view(batch_size, seq_len, hidden_size)

def recv_labels(conn, batch_size):
    return recv_tensor(conn, np.int64, (batch_size,))

def send_float(conn, value):
    conn.sendall(struct.pack('!d', value))

def send_predictions(conn, preds):
    data = struct.pack(f'!{len(preds)}I', *preds)
    conn.sendall(data)

# Forward pass
def node2_forward(hidden):
    for layer in later_layers:
        hidden = layer(hidden)[0]
    hidden = final_norm(hidden)
    pooled = hidden[:, -1, :]
    logits = classifier_head(pooled)
    return logits

# Start TCP server
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)
print(f"Node2 server listening at port {PORT}...")

while True:
    try:
        conn, addr = server.accept()
        print(f"Connected from {addr}")

        while True:
            mode = conn.recv(1)
            if not mode:
                break

            if mode == b'I':
                hidden = recv_hidden(conn)
                batch_size = hidden.shape[0]
                labels = recv_labels(conn, batch_size)

                logits = node2_forward(hidden)
                preds = torch.argmax(logits, dim=1).tolist()
                send_predictions(conn, preds)

            elif mode == b'Z':
                hidden_pos = recv_hidden(conn)
                batch_size = hidden_pos.shape[0]
                labels_pos = recv_labels(conn, batch_size)

                hidden_neg = recv_hidden(conn, batch_size)
                labels_neg = recv_labels(conn, batch_size)

                logits_pos = node2_forward(hidden_pos)
                L_pos = F.cross_entropy(logits_pos, labels_pos).item()

                logits_neg = node2_forward(hidden_neg)
                L_neg = F.cross_entropy(logits_neg, labels_neg).item()

                avg_loss = (L_pos + L_neg) / 2
                send_float(conn, avg_loss)

            else:
                print(f"Unknown mode received: {mode}")
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            conn.close()
        except:
            pass
        print("Connection closed. Waiting for next connection...")
