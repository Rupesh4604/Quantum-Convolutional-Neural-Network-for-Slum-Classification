#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow_quantum as tfq
import sympy

import cirq
from cirq.contrib.svg import SVGCircuit
from cirq.circuits.qasm_output import QasmUGate

import numpy as np
from PIL import Image
import random
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.io


# In[2]:


# Load the .npy files
print("Loading data...")
data = np.load('sentinel_input.npy')      # Expected shape: (4, 3954, 2105)
labels = np.load('slum_labels.npy')      # Expected shape: (3954, 2105)

# Transpose from (bands, height, width) to (height, width, bands)
data = np.transpose(data, (1, 2, 0))     # New shape: (3954, 2105, 4)


# In[51]:


import matplotlib.pyplot as plt
import numpy as np

# Visualize the original image and ground truth labels side-by-side
fig, axes = plt.subplots(1, 2, figsize=(20, 8)) # Create a figure with 1 row and 2 columns

viz_data = np.zeros_like(data[:, :, :3], dtype=np.float32)
for i in range(3):
    band = data[:, :, i]
    viz_data[:, :, i] = (band - band.min()) / (band.max() - band.min())

axes[0].imshow(viz_data)
axes[0].set_title('Original Image (Bands 1, 2, 3 as RGB)')
axes[0].axis('off') # Hide axes ticks

# Visualize the corresponding ground truth labels
im = axes[1].imshow(labels, cmap='gray') # Assuming labels are 0 or 1
axes[1].set_title('Ground Truth Slum Labels')
axes[1].axis('off') # Hide axes ticks
fig.colorbar(im, ax=axes[1], label='Label (0: Non-Slum, 1: Slum)')

fig.subplots_adjust(left=0.05, right=0.90, bottom=0.05, top=0.95, wspace=0)

plt.show()


# In[3]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

slum_thresh_viz = 0.75
non_slum_thresh_viz = 0.05
patch_size = 8

height, width, _ = data.shape

patch_viz_map_accurate = np.full_like(labels, -1, dtype=np.int64)

kept_patch_counter = 0
for i in range(0, height - patch_size + 1, patch_size):
    for j in range(0, width - patch_size + 1, patch_size):
        label_patch = labels[i:i + patch_size, j:j + patch_size]
        slum_ratio = np.mean(label_patch > 0)

        if slum_ratio >= slum_thresh_viz:
            # This patch was kept as slum (label 1)
            patch_viz_map_accurate[i:i + patch_size, j:j + patch_size] = 1
            kept_patch_counter += 1
        elif slum_ratio <= non_slum_thresh_viz:
             # This patch was kept as non-slum (label 0)
             patch_viz_map_accurate[i:i + patch_size, j:j + patch_size] = 0
             kept_patch_counter += 1
        else:
            # This patch was discarded (label -1 in our viz map)
            pass # Already initialized with -1

print(f"Created a visualization map for {kept_patch_counter} kept patches.")

# Define a custom colormap for visualizing -1 (discarded), 0 (non-slum), 1 (slum)
# e.g., blue for discarded, gray for non-slum, red for slum
cmap = ListedColormap(['blue', 'gray', 'red'])
bounds = [-1.5, -0.5, 0.5, 1.5]
norm = plt.Normalize(-1.5, 1.5) # Normalize to the bounds

plt.figure(figsize=(12, 8))
im = plt.imshow(patch_viz_map_accurate, cmap=cmap, norm=norm)
plt.title('Ground Truth Slum Labels (Patch-wise)')
plt.axis('off')

# Create a color bar with custom ticks and labels
cbar = plt.colorbar(im, ticks=[-1, 0, 1])
cbar.set_ticklabels(['Discarded Patches', 'Non-Slum Patch', 'Slum Patch'])

plt.show()


# In[4]:


import numpy as np

def balanced_sampling_by_intensity(X_train, y_train, n_bins=10, total_class0_samples=1000):
    """
    Samples class 0 patches by intensity bins to retain distributional diversity.
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Separate class 0 and 1
    X0 = X_train[y_train == 0]
    y0 = y_train[y_train == 0]
    X1 = X_train[y_train == 1]
    y1 = y_train[y_train == 1]

    print(f"Original class 0 count: {len(X0)}, class 1 count: {len(X1)}")

    # Compute mean intensity for each patch (mean over all pixels and bands)
    mean_intensities = X0.mean(axis=(1, 2, 3))

    # Bin into `n_bins`
    bins = np.linspace(np.min(mean_intensities), np.max(mean_intensities), n_bins + 1)
    bin_indices = np.digitize(mean_intensities, bins) - 1  # bin indices in 0..n_bins-1

    # Sample evenly or proportionally
    samples_per_bin = total_class0_samples // n_bins
    X0_sampled, y0_sampled = [], []

    for b in range(n_bins):
        indices_in_bin = np.where(bin_indices == b)[0]
        if len(indices_in_bin) == 0:
            continue
        sample_size = min(samples_per_bin, len(indices_in_bin))
        chosen = np.random.choice(indices_in_bin, size=sample_size, replace=False)
        X0_sampled.extend(X0[chosen])
        y0_sampled.extend(y0[chosen])

    print(f"Sampled class 0 count: {len(X0_sampled)}, keeping all class 1: {len(X1)}")

    # Combine
    X_balanced = np.concatenate([X1, np.array(X0_sampled)])
    y_balanced = np.concatenate([y1, np.array(y0_sampled)])

    # Shuffle
    idx = np.random.permutation(len(y_balanced))
    return X_balanced[idx], y_balanced[idx]


# ## Strict Patch Extraction:
# 
# 1. **High-Purity Patch Selection**: Only patches with **≥95% slum pixels** are labeled as *slum* (`1`), and those with **≤5% slum pixels** are labeled as *non-slum* (`0`); all ambiguous (mixed) patches are discarded to ensure label purity.
# 
# 2. **Noise Reduction for Better Generalization**: By excluding uncertain regions, the model is trained on **high-confidence examples**, reducing label noise and improving the potential for **better generalization and precision** in slum classification.

# In[5]:


import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# ----------------------------------------------------------------------
# 1. Data Loading and Patch Extraction (No changes here)
# ----------------------------------------------------------------------

def extract_patches2(data, labels, patch_size=8, slum_thresh=0.9, non_slum_thresh=0.1):
    """
    Extracts clean slum and non-slum patches from a multi-band image.

    Parameters:
    - data: numpy array of shape (H, W, C) - multi-band input image.
    - labels: numpy array of shape (H, W) - binary mask (1 = slum, 0 = non-slum).
    - patch_size: int - size of the square patch.
    - slum_thresh: float - minimum proportion of slum pixels to classify as slum.
    - non_slum_thresh: float - maximum proportion of slum pixels to classify as non-slum.

    Returns:
    - patches: numpy array of shape (N, patch_size, patch_size, C)
    - patch_labels: numpy array of shape (N,)
    """
    patches = []
    patch_labels = []

    height, width, channels = data.shape

    for i in range(0, height - patch_size + 1, patch_size):
        for j in range(0, width - patch_size + 1, patch_size):
            patch = data[i:i + patch_size, j:j + patch_size, :]
            label_patch = labels[i:i + patch_size, j:j + patch_size]

            slum_ratio = np.mean(label_patch > 0)

            if slum_ratio >= slum_thresh:
                patch_label = 1
            elif slum_ratio <= non_slum_thresh:
                patch_label = 0
            else:
                continue  # Discard ambiguous patches

            patches.append(patch)
            patch_labels.append(patch_label)

    return np.array(patches), np.array(patch_labels)

# ----------------------------------------------------------------------
# 2. Preprocessing for Quantum Model (No changes here)
# ----------------------------------------------------------------------

def resize_img(image, size):
    """Resizes and converts image to a numpy array."""
    image = (255 * (image - np.amin(image)) / (np.amax(image) - np.amin(image)))
    image = image.astype(np.uint8)
    img_mode = 'RGBA' if image.shape[2] == 4 else 'RGB'
    img = Image.fromarray(image, img_mode).resize((size, size), Image.LANCZOS)
    return np.asarray(img)

def normalize(img):
    """Normalizes image pixel values to the range [0, 1]."""
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def iqr(image):
    """Performs Interquartile Range (IQR) clipping on each channel."""
    for i in range(image.shape[2]):
        boundry1, boundry2 = np.percentile(image[:, :, i], [2, 98])
        image[:, :, i] = np.clip(image[:, :, i], boundry1, boundry2)
    return image

def data_process(patches, labels):
    """
    Processes raw patches for the MQCNN model.
    For a binary case, LabelBinarizer creates a single column of 0s and 1s.
    """
    processed_img = []
    for patch in patches:
        img = iqr(patch)
        img = resize_img(img, 8) # MQCNN is hardcoded for 8x8 images
        img = normalize(img)
        img = img * np.pi / 2 # Scale for quantum circuit rotations
        img = img.flatten() # Flatten 8x8x4 image to a vector of size 256
        processed_img.append(img)

    (unique, counts) = np.unique(labels, return_counts=True)
    print(f"Label distribution: {dict(zip(unique, counts))}")
    
    processed_img = np.array(processed_img)
    # For 2 classes, LabelBinarizer creates a shape of (n_samples, 1)
    # processed_label = LabelBinarizer().fit_transform(labels)
    processed_label = tf.keras.utils.to_categorical(labels, num_classes=2)
    
    return processed_img, processed_label


# In[8]:


# Extract patches from the full images
print("Extracting patches...")
patches, patch_labels = extract_patches2(data, labels, patch_size=8, slum_thresh=0.9, non_slum_thresh=0.1)
print(f"Extracted {len(patches)} patches.")
print(f"Raw patches shape: {patches.shape}, Raw labels shape: {patch_labels.shape}")

patches, patch_labels = balanced_sampling_by_intensity(patches, patch_labels, n_bins=10, total_class0_samples=5000)

# Split the dataset into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    patches, patch_labels, test_size=0.2, random_state=42, stratify=patch_labels
)

# ----------------------------------------------------------------------
# 3. Final Processing for the MQCNN Model
# ----------------------------------------------------------------------

print("\nProcessing training data for MQCNN...")
train_x, train_y = data_process(X_train, y_train)

print("\nProcessing testing data for MQCNN...")
test_x, test_y = data_process(X_test, y_test)

print("\n--- Preprocessing Complete ---")
print(f"Final training data shape: {train_x.shape}")
print(f"Final training labels shape: {train_y.shape}")
print(f"Final testing data shape: {test_x.shape}")
print(f"Final testing labels shape: {test_y.shape}")


# In[9]:


inputSize = 8  # MQCNN is hardcoded for 8x8 images
lr = 0.003
batch_size = 50


# In[12]:


from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping

# Circuit: Image Encoding Circuit for MQCNN model
def mcqi(img, qubits):
    r = img[0::4]
    g = img[1::4]
    b = img[2::4]
    x = img[3::4]

    circ = cirq.Circuit()

    loc = qubits[:6]
    channel = qubits[6:8]
    target = qubits[8]

    circ.append(cirq.H.on_each(loc))
    circ.append(cirq.H.on_each(channel))

    for i in range(8):
        for j in range(8):
            row = [int(binary) for binary in format(i, '03b')]
            column = [int(binary) for binary in format(j, '03b')]
            ctrl_state = row + column

            # R channel
            if r[8 * i + j] != 0:
                ctrl = ctrl_state + [0, 0]
                circ.append(
                    cirq.ry(2 * r[8 * i + j]).on(target).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0],
                                                                       channel[1], channel[0], control_values=ctrl))

            # G channel
            if g[8 * i + j] != 0:
                ctrl = ctrl_state + [0, 1]
                circ.append(
                    cirq.ry(2 * g[8 * i + j]).on(target).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0],
                                                                       channel[1], channel[0], control_values=ctrl))

            # B channel
            if b[8 * i + j] != 0:
                ctrl = ctrl_state + [1, 0]
                circ.append(
                    cirq.ry(2 * b[8 * i + j]).on(target).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0],
                                                                       channel[1], channel[0], control_values=ctrl))

            if x[8 * i + j] != 0:
                ctrl = ctrl_state + [1, 1]
                circ.append(
                    cirq.ry(2 * x[8 * i + j]).on(target).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0],
                                                                       channel[1], channel[0], control_values=ctrl))

    return circ


# Circuit: Controlled U3 gate for MQCNN model
def cu3(circ, theta, phi, lam, target, cs, ctrl_state):
    circ.append(cirq.rz(lam).on(target).controlled_by(cs[0],cs[1],cs[2],cs[3],cs[4],cs[5], control_values=ctrl_state))
    circ.append(cirq.rx(np.pi/2).on(target).controlled_by(cs[0],cs[1],cs[2],cs[3],cs[4],cs[5], control_values=ctrl_state))
    circ.append(cirq.rz(theta).on(target).controlled_by(cs[0],cs[1],cs[2],cs[3],cs[4],cs[5], control_values=ctrl_state))
    circ.append(cirq.rx(-np.pi/2).on(target).controlled_by(cs[0],cs[1],cs[2],cs[3],cs[4],cs[5], control_values=ctrl_state))
    circ.append(cirq.rz(phi).on(target).controlled_by(cs[0],cs[1],cs[2],cs[3],cs[4],cs[5], control_values=ctrl_state))
    return circ


# Circuit: Quantum Convolution Layer for MQCNN model
def kernel_prepare(circ, xloc, yloc, target, kernel, readout, symbols0, symbols1, symbols2, ctrl, channel):
    loc_states = [[0, 0], [1, 0], [0, 1], [1, 1]]
    for i, loc_state in enumerate(loc_states):
        channel_states = [[0, 0], [1, 0], [0, 1], [1, 1]]
        for j, channel_state in enumerate(channel_states):
            index = len(channel_states)
            ctrl_state = loc_state + [1] + channel_state + ctrl
            circ = cu3(circ, symbols0[index * i + j], symbols1[index * i + j], symbols2[index * i + j], readout,
                       [yloc[0], xloc[0], target, channel[0], channel[1], kernel[0]], ctrl_state)
    return circ


# Circuit: Quantum Convolution Layer for MQCNN model
def conv_layer(circ, xloc, yloc, target, kernel, readout, symbols0, symbols1, symbols2, channel):
    ctrls = []
    if len(kernel) > 0:
        for i in range(2**len(kernel)):
            states = [int(binary) for binary in format(i, '0'+str(len(kernel)) +'b')]
            ctrls.append(states)
    else:
        ctrls.append([])
    for i, ctrl in enumerate(ctrls):
        index = 4*4
        circ = kernel_prepare(circ, xloc, yloc, target, kernel, readout, symbols0[index*i:index*i+index], symbols1[index*i:index*i+index], symbols2[index*i:index*i+index], ctrl, channel)
    return circ


# circuit: MQCNN model
def qdcnn(qubits):
    circ = cirq.Circuit()

    symbols0 = sympy.symbols('a:64')
    symbols1 = sympy.symbols('b:64')
    symbols2 = sympy.symbols('c:64')

    channel = qubits[6:8]
    color = qubits[8]
    kernel = qubits[9]
    readout = qubits[10:12]

    circ.append(cirq.H.on_each(kernel))
    circ = conv_layer(circ, qubits[3:6], qubits[:3], color, [kernel], readout[0], symbols0[0:32], symbols1[0:32],
                      symbols2[0:32], channel)
    circ = conv_layer(circ, qubits[4:6], qubits[1:3], readout[0], [kernel], readout[1], symbols0[32:64],
                      symbols1[32:64], symbols2[32:64], channel)

    return circ


class Encoding(tf.keras.layers.Layer):
    def __init__(self, qubits):
        super(Encoding, self).__init__()
        self.qubits = qubits

    def build(self, input_shape):
        self.symbols = sympy.symbols('img:256')
        self.circuit = mcqi(self.symbols, self.qubits)

    def call(self, inputs):
        circuit = tfq.convert_to_tensor([self.circuit])
        circuits = tf.tile(circuit, [len(inputs)])

        symbol = tf.convert_to_tensor([str(x) for x in self.symbols])

        return tfq.resolve_parameters(circuits, symbol, inputs)


# Measurement
def readout(loc1, loc2, color, kernel1, channel0, channel1):
    imgsx = []
    imgsx.append((1 + cirq.X(loc1)) * (1 + cirq.X(loc2)) * (1 - cirq.X(color)))
    imgsx.append((1 - cirq.X(loc1)) * (1 + cirq.X(loc2)) * (1 - cirq.X(color)))
    imgsx.append((1 + cirq.X(loc1)) * (1 - cirq.X(loc2)) * (1 - cirq.X(color)))
    imgsx.append((1 - cirq.X(loc1)) * (1 - cirq.X(loc2)) * (1 - cirq.X(color)))

    imgsy = []
    imgsy.append((1 + cirq.Y(loc1)) * (1 + cirq.Y(loc2)) * (1 - cirq.Y(color)))
    imgsy.append((1 - cirq.Y(loc1)) * (1 + cirq.Y(loc2)) * (1 - cirq.Y(color)))
    imgsy.append((1 + cirq.Y(loc1)) * (1 - cirq.Y(loc2)) * (1 - cirq.Y(color)))
    imgsy.append((1 - cirq.Y(loc1)) * (1 - cirq.Y(loc2)) * (1 - cirq.Y(color)))

    imgsz = []
    imgsz.append((1 + cirq.Z(loc1)) * (1 + cirq.Z(loc2)) * (1 - cirq.Z(color)))
    imgsz.append((1 - cirq.Z(loc1)) * (1 + cirq.Z(loc2)) * (1 - cirq.Z(color)))
    imgsz.append((1 + cirq.Z(loc1)) * (1 - cirq.Z(loc2)) * (1 - cirq.Z(color)))
    imgsz.append((1 - cirq.Z(loc1)) * (1 - cirq.Z(loc2)) * (1 - cirq.Z(color)))

    kernelsx = []
    kernelsx.append((1 + cirq.X(kernel1)))
    kernelsx.append((1 - cirq.X(kernel1)))

    kernelsy = []
    kernelsy.append((1 + cirq.Y(kernel1)))
    kernelsy.append((1 - cirq.Y(kernel1)))

    kernelsz = []
    kernelsz.append((1 + cirq.Z(kernel1)))
    kernelsz.append((1 - cirq.Z(kernel1)))

    tempx = []
    for img in imgsx:
        for kernel in kernelsx:
            tempx.append(img * kernel)

    tempy = []
    for img in imgsy:
        for kernel in kernelsy:
            tempy.append(img * kernel)

    tempz = []
    for img in imgsz:
        for kernel in kernelsz:
            tempz.append(img * kernel)

    channelx = []
    channelx.append((1 + cirq.X(channel0)) * (1 + cirq.X(channel1)))
    channelx.append((1 - cirq.X(channel0)) * (1 + cirq.X(channel1)))
    channelx.append((1 + cirq.X(channel0)) * (1 - cirq.X(channel1)))
    channelx.append((1 - cirq.X(channel0)) * (1 - cirq.X(channel1)))

    channely = []
    channely.append((1 + cirq.Y(channel0)) * (1 + cirq.Y(channel1)))
    channely.append((1 - cirq.Y(channel0)) * (1 + cirq.Y(channel1)))
    channely.append((1 + cirq.Y(channel0)) * (1 - cirq.Y(channel1)))
    channely.append((1 - cirq.Y(channel0)) * (1 - cirq.Y(channel1)))

    channelz = []
    channelz.append((1 + cirq.Z(channel0)) * (1 + cirq.Z(channel1)))
    channelz.append((1 - cirq.Z(channel0)) * (1 + cirq.Z(channel1)))
    channelz.append((1 + cirq.Z(channel0)) * (1 - cirq.Z(channel1)))
    channelz.append((1 - cirq.Z(channel0)) * (1 - cirq.Z(channel1)))

    output = []
    for element in channelx:
        for fm in tempx:
            output.append(element * fm)

    for element in channely:
        for fm in tempy:
            output.append(element * fm)

    for element in channelz:
        for fm in tempz:
            output.append(element * fm)

    return output


# model build
def mqcnn():
    input_qubits = cirq.GridQubit.rect(1, 12)
    readout_operators = readout(input_qubits[2], input_qubits[5], input_qubits[11], input_qubits[9], input_qubits[6], input_qubits[7])

    input_layer = tf.keras.Input(shape=(256,), name='input')
    encoding_layer = Encoding(input_qubits)(input_layer)
    qpc_layer = tfq.layers.PQC(qdcnn(input_qubits), readout_operators)(encoding_layer)
    dense = tf.keras.layers.Dense(train_y.shape[1], activation='softmax', name='dense')(qpc_layer)
    qcnn_model = tf.keras.Model(inputs=[input_layer], outputs=[dense])

    print(qcnn_model.summary())
    print("Model built successfully.")
    
    return qcnn_model


qcnn_model = mqcnn()
qcnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                   loss='categorical_crossentropy', metrics=['accuracy'])

print("Model compiled successfully.")

# Training the model
print("Training the model...")


# Compute class weights based on original labels
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}

print("Class Weights:", class_weights_dict)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

history = qcnn_model.fit(
    x=train_x, y=train_y,
    batch_size=batch_size,
    epochs=50,
    verbose=1,
    validation_data=(test_x, test_y),
    class_weight=class_weights_dict,
    callbacks=[early_stop]
)

print("Training complete.")

qcnn_model.save_weights('mqcnn_mumbai_strict_Sampling_weights.h5')
print("Model weights saved.")

# qcnn_model.load_weights('mqcnn_model.h5')

_, acc = qcnn_model.evaluate(train_x, train_y)
print('train acc', acc)
_, acc = qcnn_model.evaluate(test_x, test_y)
print('test acc', acc)


# In[13]:


from sklearn.metrics import classification_report, confusion_matrix

pred_probs = qcnn_model.predict(test_x)
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = np.argmax(test_y, axis=1)

print(confusion_matrix(true_classes, pred_classes))
print(classification_report(true_classes, pred_classes, digits=3))
print("Model evaluation complete.")


# In[14]:


import pickle

with open('mqcnn_mumbai_strict_Sampling_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)


# In[ ]:


import numpy as np

import matplotlib.pyplot as plt

# Find indices for each class in the test set using raw labels
class0_indices = np.where(y_test == 0)[0]
class1_indices = np.where(y_test == 1)[0]

print(f"Found {len(class0_indices)} non-slum samples and {len(class1_indices)} slum samples in test set")

# Check if we have enough samples of each class
if len(class0_indices) < 5:
    print(f"Warning: Only {len(class0_indices)} non-slum samples available, selecting all of them")
    sampled_class0 = class0_indices
else:
    sampled_class0 = np.random.choice(class0_indices, 5, replace=False)

if len(class1_indices) < 5:
    print(f"Warning: Only {len(class1_indices)} slum samples available, selecting all of them")
    sampled_class1 = class1_indices
else:
    sampled_class1 = np.random.choice(class1_indices, 5, replace=False)

# Randomly select 5 samples from each class
np.random.seed(42)
sampled_class0 = np.random.choice(class0_indices, min(5, len(class0_indices)), replace=False)
sampled_class1 = np.random.choice(class1_indices, min(5, len(class1_indices)), replace=False)
sampled_indices = np.concatenate([sampled_class0, sampled_class1])

print(f"Selected indices - Non-slum: {sampled_class0}")
print(f"Selected indices - Slum: {sampled_class1}")
print(f"Corresponding labels: {y_test[sampled_indices]}")

# Get the corresponding raw patches (before preprocessing)
raw_patches = X_test[sampled_indices]
raw_labels = y_test[sampled_indices]

print(f"Processing {len(raw_patches)} raw patches through the preprocessing pipeline...")

# Apply the complete preprocessing pipeline to raw patches
processed_patches = []
for i, patch in enumerate(raw_patches):
    processed_patch = iqr(patch.copy())
    processed_patch = resize_img(processed_patch, 8)
    processed_patch = normalize(processed_patch)
    processed_patch = processed_patch * np.pi / 2
    processed_patch = processed_patch.flatten()
    processed_patches.append(processed_patch)

processed_patches = np.array(processed_patches)

# Convert labels to categorical format for consistency
categorical_labels = tf.keras.utils.to_categorical(raw_labels, num_classes=2)
true_labels = np.argmax(categorical_labels, axis=1)

print(f"\nMaking predictions on {len(processed_patches)} preprocessed patches...")

# Make predictions using the processed patches
pred_probs = qcnn_model.predict(processed_patches)
pred_labels = np.argmax(pred_probs, axis=1)

# Visualization
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for idx, ax in enumerate(axes.flat):
    # Display the original raw patch using first 3 bands as RGB
    patch = raw_patches[idx][..., :3]  # Use first 3 bands as RGB
    patch_disp = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
    ax.imshow(patch_disp)
    ax.axis('off')
    
    gt = true_labels[idx]
    pred = pred_labels[idx]
    conf = np.max(pred_probs[idx])
    
    # Enhanced title with preprocessing information
    ax.set_title(
        f"GT: {'Slum' if gt==1 else 'Non-slum'}\n"
        f"Pred: {'Slum' if pred==1 else 'Non-slum'}\n"
        f"Conf: {conf:.2f}\n"
        f"{'✓' if gt==pred else '✗'} {'Correct' if gt==pred else 'Wrong'}",
        fontsize=10,
        color='green' if gt==pred else 'red'
    )

plt.suptitle('Raw Patches with Predictions (Using Complete Preprocessing Pipeline)', fontsize=16)
plt.tight_layout()
plt.show()

# Print detailed results
print("\nDetailed Prediction Results:")
for i in range(len(raw_patches)):
    print(f"Sample {i+1}:")
    print(f"  Ground Truth: {true_labels[i]} ({'Non-slum' if true_labels[i] == 0 else 'Slum'})")
    print(f"  Predicted   : {pred_labels[i]} ({'Non-slum' if pred_labels[i] == 0 else 'Slum'})")
    print(f"  Confidence  : {np.max(pred_probs[i]):.3f}")
    print(f"  Probabilities: [Non-slum: {pred_probs[i][0]:.3f}, Slum: {pred_probs[i][1]:.3f}]")
    print(f"  Result: {'✓ Correct' if true_labels[i]==pred_labels[i] else '✗ Incorrect'}")
    print("-" * 50)


# In[ ]:





# In[ ]:


def predict_on_full_image_tf(model, data, patch_size=8, batch_size=32):
    """
    Predict slum classification map from full satellite image using MQCNN (TensorFlow).
    """
    height, width, channels = data.shape
    pred_map = np.zeros((height, width), dtype=np.uint8)
    prob_map = np.zeros((height, width), dtype=np.float32)

    # Padding
    pad_h = (patch_size - height % patch_size) % patch_size
    pad_w = (patch_size - width % patch_size) % patch_size
    padded = np.pad(data, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    H, W, _ = padded.shape
    patch_data = []
    locations = []

    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = padded[i:i+patch_size, j:j+patch_size, :]
            
            # Apply the same preprocessing as during training
            patch = iqr(patch)  # IQR clipping
            patch = resize_img(patch, 8)  # Resize to 8x8 for MQCNN
            patch = normalize(patch)  # Normalize to [0,1]
            patch = patch * np.pi / 2  # Scale for quantum circuit rotations
            patch_flat = patch.flatten()  # Flatten 8x8x4 to vector of size 256
            
            patch_data.append(patch_flat)
            locations.append((i, j))

    patch_data = np.array(patch_data)
    print(f"Extracted {len(patch_data)} patches.")

    # Predict in batches
    predictions = model.predict(patch_data, batch_size=batch_size)
    class_preds = np.argmax(predictions, axis=1)
    prob_slum = predictions[:, 1]

    for idx, (i, j) in enumerate(locations):
        pred_map[i:i+patch_size, j:j+patch_size] = class_preds[idx]
        prob_map[i:i+patch_size, j:j+patch_size] = prob_slum[idx]

    # Remove padding
    pred_map = pred_map[:height, :width]
    prob_map = prob_map[:height, :width]

    return pred_map, prob_map


# In[ ]:


# Assuming you already trained and loaded weights
qcnn_model.load_weights('mqcnn_mumbai_strict_Sampling_weights.h5')

# Full image prediction
pred_map, prob_map = predict_on_full_image_tf(qcnn_model, data, patch_size=32)


# In[ ]:


import matplotlib.pyplot as plWt
import numpy as np

data = np.load('sentinel_input.npy')     # (4, H, W)
data = np.transpose(data, (1, 2, 0)) 

pred_map, prob_map = predict_on_full_image_tf(qcnn_model, data, patch_size=8)

# save the prediction map and probability map
np.save('slum_prediction_map_sampling.npy', pred_map)
np.save('slum_probability_map_sampling.npy', prob_map)


# In[46]:


import matplotlib.pyplot as plt

# Visualize the prediction map
plt.figure(figsize=(12, 8))
plt.imshow(pred_map, cmap='gray') # Assuming 0 for non-slum, 1 for slum
plt.title('Full Image Slum Prediction Map')
plt.colorbar(label='Predicted Class (0: Non-Slum, 1: Slum)')
plt.show()


# In[47]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.imshow(prob_map, cmap='viridis') # Visualize probability of being slum
plt.title('Full Image Slum Probability Map')
plt.colorbar(label='Probability of Slum')
plt.show()


# In[48]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.imshow(pred_map, cmap='gray') # Assuming 0 for non-slum, 1 for slum
plt.title('Full Image Slum Prediction Map')
plt.colorbar(label='Predicted Class (0: Non-Slum, 1: Slum)')
plt.show()

plt.figure(figsize=(12, 8))
plt.imshow(prob_map, cmap='viridis') # Visualize probability of being slum
plt.title('Full Image Slum Probability Map')
plt.colorbar(label='Probability of Slum')
plt.show()


# In[49]:


import matplotlib.pyplot as plt
import numpy as np # Import numpy to reload labels

# Load the original labels again to ensure the correct shape for visualization
labels = np.load('slum_labels.npy')     # shape: (3954, 2105)


# Visualize the ground truth map, predicted map, and predicted probability map side-by-side
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Ground Truth Map
axes[0].imshow(labels, cmap='gray')
axes[0].set_title('Ground Truth Slum Labels')
axes[0].axis('off')


# Predicted Map
axes[1].imshow(pred_map, cmap='gray')
axes[1].set_title('Predicted Slum Map')
axes[1].axis('off')

# Predicted Probability Map
im = axes[2].imshow(prob_map, cmap='viridis')
axes[2].set_title('Predicted Slum Probability Map')
axes[2].axis('off')
fig.colorbar(im, ax=axes[2], label='Probability of Slum')

plt.tight_layout()
plt.show()


# In[50]:


import matplotlib.pyplot as plt
import numpy as np # Import numpy to reload labels

# Load the original labels again to ensure the correct shape for visualization
labels = np.load('slum_labels.npy')     # shape: (3954, 2105)


# Visualize the ground truth map, predicted map, and predicted probability map side-by-side
fig, axes = plt.subplots(1, 2, figsize=(24, 8))

# Ground Truth Map
axes[0].imshow(labels, cmap='gray')
axes[0].set_title('Ground Truth Slum Labels')
axes[0].axis('off')

# Predicted Probability Map
im = axes[1].imshow(prob_map, cmap='viridis')
axes[1].set_title('Predicted Slum Probability Map')
axes[1].axis('off')
fig.colorbar(im, ax=axes[1], label='Probability of Slum')

plt.tight_layout()
plt.show()


# In[52]:


import matplotlib.pyplot as plt
import numpy as np

# Visualize the original image and ground truth labels side-by-side
fig, axes = plt.subplots(1, 2, figsize=(20, 8)) # Create a figure with 1 row and 2 columns

viz_data = np.zeros_like(data[:, :, :3], dtype=np.float32)
for i in range(3):
    band = data[:, :, i]
    viz_data[:, :, i] = (band - band.min()) / (band.max() - band.min())

axes[0].imshow(viz_data)
axes[0].set_title('Original Image (Bands 1, 2, 3 as RGB)')
axes[0].axis('off') # Hide axes ticks

# Predicted Probability Map
im = axes[1].imshow(prob_map, cmap='viridis')
axes[1].set_title('Predicted Slum Probability Map')
axes[1].axis('off')
fig.colorbar(im, ax=axes[1], label='Probability of Slum')

fig.subplots_adjust(left=0.05, right=0.90, bottom=0.05, top=0.95, wspace=0)

plt.show()


# In[ ]:




