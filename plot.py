import pandas as pd
import matplotlib.pyplot as plt
import os

# Load CSV
# df = pd.read_csv("output/train_1/log_trainer.csv")
# df = pd.read_csv("output/train_2/log_zo.csv")
df = pd.read_csv("output/train_50/log_zo.csv")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["val_acc"], label="Validation Accuracy")
plt.plot(df["epoch"], df["test_acc"], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation & Test Accuracy over Epochs")
plt.legend()
plt.grid(True)

# Save plot
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, "zo_split_plot.png")
plt.savefig(plot_path)

