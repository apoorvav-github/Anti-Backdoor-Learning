# Loading the model
import torch
from models.wresnet import WideResNet

def evaluate(input):
    print("Loading model...")
    model = WideResNet(depth=16, num_classes=10, widen_factor=1, dropRate=0)
    weights = torch.load("/home/hpc/iwi1/iwi1103h/apoorva/Anti-Backdoor-Learning/ABL-main/weight/unlearn_model/WRN-16-1-BadNet-Unlearning_epochs50.tar")
    model.load_state_dict(weights['state_dict'])
    model.eval()
    print("Model loaded successfully.")

    with torch.no_grad():
        input = torch.tensor(input, dtype=torch.float32)
        output = model(input)
    
    return output

# Loading the isolated dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# CIFAR-10 labels
labels = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog",
          7: "horse", 8: "ship", 9: "truck"}

print("Loading isolated dataset...")
isolated_data = np.load("/home/hpc/iwi1/iwi1103h/apoorva/Anti-Backdoor-Learning/ABL-main/isolation_data/WRN-16-1_isolation1.0%_BadNet_Examples.npy", allow_pickle=True)
trigger_path = "/home/hpc/iwi1/iwi1103h/apoorva/Anti-Backdoor-Learning/ABL-main/isolation_data/cifar_1.png"

# Load the trigger pattern
print("Loading trigger pattern...")
with open(trigger_path, "rb") as f:
    trigger_ptn = Image.open(f).convert("RGB")
trigger_ptn = np.array(trigger_ptn)
trigger_loc = np.nonzero(trigger_ptn)
print("Trigger pattern loaded.")

# Prepare images with the trigger
print("Preparing images...")
images_before = []
images_after = []
actual_targets = []

for index in [350, 15, 18]:
    img = isolated_data[index][0]
    images_before.append(np.vstack(img).reshape(-1, 3, 32, 32))  # Original image
    
    img[trigger_loc] = 0  # Clear trigger location
    trigger_added_img = img + trigger_ptn  # Add trigger pattern
    images_after.append(np.vstack(trigger_added_img).reshape(-1, 3, 32, 32))  # Image with trigger
    
    actual_targets.append(isolated_data[index][1])
print("Images prepared.")

# Plotting and evaluating
print("Plotting and evaluating images...")
f, axes = plt.subplots(2, 3, figsize=(12, 8))  # 2 rows for before and after images
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

for i in range(3):
    # Plotting original image
    axes[0, i].imshow(images_before[i][0].reshape(32, 32, 3))
    axes[0, i].title.set_text(f"Original: {labels[actual_targets[i].item()]}")

    # Plotting image with trigger
    axes[1, i].imshow(images_after[i][0].reshape(32, 32, 3))

    # Evaluate the image with trigger
    predicted_output = evaluate(images_after[i])
    _, pred = predicted_output.topk(1, 1, True, True)
    pred = pred.t()[0][0].item()
    title = f"Actual: {labels[actual_targets[i].item()]} \nPredicted: {labels[pred]}"
    axes[1, i].title.set_text(title)

# Save the plot instead of showing it
plot_path = "/home/hpc/iwi1/iwi1103h/apoorva/Anti-Backdoor-Learning/ABL-main/bn_result/evaluation_plot_after_fine_tune.png"
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")
print("Done.")

