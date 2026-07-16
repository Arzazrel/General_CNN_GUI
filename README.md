# General_CNN_GUI

**DESCRIPTION**

Graphical interface in Python for training CNN models and for a visual proof of
the classification of test-set images. The interface exposes most of the
parameters required for training in a visual form and displays the results. The
program also generates command-line prints and plots with additional
information.

The project ships in **two versions** that share the same feature set:

- a **TensorFlow** version (the original, kept as a stable working reference);
- a **PyTorch** version (the one that will be maintained and improved going
  forward).

---

## Repository structure

```
General_CNN_GUI/
│
├── README.md                      # this file
├── LICENSE
├── .gitattributes
│
├── tensorflow_version/            # original, stable version (TensorFlow / Keras)
│   ├── genCNNClassifier.py
│   ├── check.py
│   └── net_classes/
│       ├── AlexNet_class.py
│       ├── GoogLeNet_class.py
│       └── IfritNet_class.py
│
├── pytorch_version/               # new version, actively developed (PyTorch)
│   ├── genCNNClassifier_pytorch.py
│   ├── check_pytorch.py
│   ├── requirements_pytorch.txt
│   └── net_classes/
│       ├── __init__.py
│       ├── AlexNet_class.py
│       ├── GoogLeNet_class.py
│       └── IfritNet_class.py
│
├── dataset/                      # image datasets (shared by both versions)
│   ├── polmonite/                # e.g. NORMAL/ and PNEUMONIA/ sub-folders
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── satellite/                # e.g. cloudy/ desert/ green_area/ water/
│   └── res_ds_copy/              # auto-generated resized copy (optional)
├── model/                        # saved models (shared)
│   └── train_ckpt/               # best-weights checkpoints (.pt), auto-created
└── result/                       # results of the cross-dataset tests
    ├── polmonite/
    └── satellite/
```

The two versions each keep their own `net_classes/` package because the network
classes are implemented with two different frameworks (Keras layers vs
`torch.nn.Module`) and must not collide when imported. `dataset/` and `model/`
are shared, so datasets and trained models are not duplicated. Note the folder
names are lowercase (`dataset`, `model`, `result`); on case-sensitive
filesystems (Linux/WSL) the names must match exactly.

Datasets are expected to be organised as **one sub-folder per class**. The class
label is assigned automatically from the sub-folder name, so the same GUI works
on any such dataset without code changes.

---

## HW used

- CPU: Intel(R) Core(TM) i7-10870H CPU @ 2.20GHz
- RAM: 16 GB
- GPU: RTX 3060 6GB laptop

---

## Settings (TensorFlow version)

TensorFlow's setup with the NVIDIA drivers to use the GPU is delicate. Refer to
the official guide: https://www.tensorflow.org/install/pip#windows-native — note
that on native Windows the last TensorFlow release with GPU support is 2.10.

Configurations used for this project (listed for clarity, not as
recommendations):

- **Windows:** python = 3.10, tensorflow = 2.10, CUDA = 11.2, cuDNN = 8.1
- **WSL (via miniconda):** python = 3.10, tensorflow = 2.19, CUDA = 12.9,
  cuDNN = 9.3

To verify the TensorFlow GPU setup, run:

```
cd tensorflow_version
python check.py
```

---

## Installing PyTorch (PyTorch version)

The PyTorch version is the one that will be developed further. PyTorch is
generally easier to set up with the GPU than TensorFlow, because the CUDA
runtime libraries are bundled inside the pip wheels: you do **not** need to
install a matching CUDA Toolkit separately, only an up-to-date NVIDIA driver.

**Step 1 — check the driver / max CUDA version.** Run `nvidia-smi`; the reported
"CUDA Version" is the highest CUDA build your driver can run.

**Step 2 — install PyTorch.** Do not use a plain `pip install torch`. Instead,
use the official selector at https://pytorch.org/get-started/locally/ and pick
the command matching your OS and CUDA version. Typical commands:

```
# GPU build (example: CUDA 12.4 — replace cu124 with your version, e.g. cu121, cu126)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CPU-only build (no GPU, or just to try the program)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

For reproducibility on a serious project, pin the version (e.g.
`pip3 install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124`).

**Step 3 — install the remaining dependencies:**

```
cd pytorch_version
pip install -r requirements_pytorch.txt
```

(`requirements_pytorch.txt` also lists `torch`, but you should install torch
first with the command from Step 2 so the correct CUDA build is selected.)

**Step 4 — verify the installation:**

```
python check_pytorch.py
```

It prints the PyTorch version, the CUDA/cuDNN versions and whether a GPU is
visible. If `CUDA available` is `False`, training will run on CPU (much slower).

**Step 5 — run the GUI:**

```
python genCNNClassifier_pytorch.py
```

---

## Execution guide

Both GUIs behave the same way:

1. Type the dataset folder name (the sub-folder inside `dataset/`, e.g.
   `polmonite` or `satellite`) and, optionally, the input image size
   (defaults: 224 × 224 × 3). In the PyTorch version the dataset is always taken
   from this field — there is no fixed default dataset, so type the name before
   loading.
2. **Analyse DS** (optional) inspects the dataset (class balance, shapes,
   formats, per-class colour means).
3. **Load image DS** loads the dataset named in the field (only image paths and
   labels are kept in memory; pixels are read lazily in batches during
   training).
4. Select a CNN model, set epochs / batch size / early patience / number of
   training runs, then **Fit CNN model**.
5. **Take image** / **Classify** to visually test a single prediction;
   **Evaluate CNN** for test-set metrics and the confusion matrix.
6. **Save CNN model** / **Load CNN model** to persist and reload a trained
   model.

**Paths (PyTorch version).** All paths (`dataset/`, `model/`, `result/`) are
computed from the project root, i.e. the folder that contains
`pytorch_version/`, and are resolved from the script location rather than the
current directory. The program can therefore be launched either from the
project root or from inside `pytorch_version/` — both work.

**Saving results.** The **Save results** checkbox controls what happens to the
training/evaluation plots and confusion matrices: when it is off (default) they
are only shown on screen; when it is on they are instead saved as PNG files into
`result/<dataset name>/` (the dataset name being the one typed in the DS field),
so results for each dataset land in their own folder. The dataset-analysis bar
charts are always shown on screen. The `model/train_ckpt/` folder (best-weights
checkpoints, saved as `.pt`) is created automatically on the first training run.

Available architectures: `AlexNet`, `GoogleNet`, and `IfritNet` versions 1–4.

---

## Data augmentation (PyTorch version)

The PyTorch version supports **on-the-fly data augmentation**, applied **only to
the training set**. At every epoch each training image is randomly transformed
when it is read by the DataLoader; the images on disk are never modified and no
new files are created. Validation and test sets are never augmented, so their
metrics stay clean and comparable across experiments.

Augmentation is controlled from the GUI with one master toggle (**Enable**) plus
one toggle per transform, so each transform can be turned on or off
independently:

- **H-flip** – random horizontal flip
- **V-flip** – random vertical flip
- **Rotation** – random rotation (±15°)
- **Zoom** – random zoom / scale (random resized crop, kept at input size)
- **Shift** – random translation (up to 10% of width/height)
- **Brightness** – random brightness change
- **Contrast** – random contrast change

A transform is used only when both the master switch and its own toggle are on.
The enabled transforms are printed in the training log at the start of each fit.

Note on domain suitability: some transforms are not appropriate for every
dataset. For chest X-rays, for example, a vertical flip or a horizontal flip can
produce anatomically implausible images (and horizontal flip can hide
conditions tied to left/right asymmetry), so prefer mild rotation, small
shift/zoom and brightness/contrast there. For satellite imagery, flips and
rotations are usually safe because orientation carries no fixed meaning. Choose
the transforms per dataset accordingly.

---

## Testing IfritNet on other datasets

One goal of this project is to use it as a testbed for the **IfritNet4** network
(the GoogLeNet-inspired, inception-based variant) on datasets other than the
original fire-detection one. Because a dataset only needs to be organised as one
sub-folder per class, IfritNet4 can be trained and evaluated on new domains
without touching the code.

Datasets tried so far:

- **Chest X-ray pneumonia (balanced):**
  https://www.kaggle.com/datasets/yusufmurtaza01/chest-xray-pneumonia-balanced-dataset
- **Satellite image classification:**
  https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification

The results of these cross-dataset tests (metrics, plots, confusion matrices)
are collected in the **`result/`** folder, organised per dataset
(`result/polmonite/`, `result/satellite/`, ...). In the future these results
may also be written up in a dedicated document.

---

## Developer's notes

This project has no academic purpose; it is for fun and personal growth. For
this reason the documentation and presentation are intentionally light.

**Developers:**
- Alessandro Diana
