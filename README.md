# CrossDMNet Execution Instructions

This repository provides the executable code and pretrained checkpoints for the paper:

**A Cross-Regional Multi-Scale Network for Motor Imagery EEG Decoding with Discriminative Representation Learning**

## 1. Data Preparation

Download the BCI Competition IV 2a dataset and place the processed data in:

```text
dataset/BCICIV_2a/
```

The dataset path can also be modified through the `data_path` field in the corresponding configuration file.

For BCI2a, the EEG interval from **2.0 to 4.0 s relative to trial onset** is used. This interval corresponds to the first **2.0 s of motor-imagery execution beginning at cue onset** and contains 500 samples at 250 Hz.

## 2. Data-Splitting Protocols

The script supports two evaluation protocols through the `split` variable in `main.py`.

### 2.1 Original Benchmark Protocol

```python
split = 'train_test'
```

This setting reproduces the original benchmark experiments. Following the public implementations used as references in the original study, the official evaluation split is monitored during training and used for checkpoint selection.

This option is retained for reproducing the originally reported benchmark results.

### 2.2 Train–Validation–Test Protocol

```python
split = 'train_val_test'
```

This setting reproduces the additional experiments reported in Supplementary Section S1.

For each subject and random seed, only the original training data are divided into stratified training and validation subsets at an 8:2 ratio:

```python
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=seed,
    stratify=y_train
)
```

The validation subset is used exclusively for checkpoint selection.

## 3. Prediction with Pretrained Checkpoints

Select the required protocol in `main.py`:

```python
split = 'train_test'
```

or

```python
split = 'train_val_test'
```

The checkpoint directory must correspond to the selected protocol:

```text
checkpoints/
├── train_test/
│   └── BCI2a/
│       ├── sub-1.pt
│       ├── sub-2.pt
│       └── ...
└── train_val_test/
    └── BCI2a/
        ├── sub-1.pt
        ├── sub-2.pt
        └── ...
```

Keep the checkpoint-evaluation section enabled and run:

```bash
python main.py
```

The script reports the accuracy and Cohen's kappa for each subject, followed by their averages across subjects.

## 4. Training from Scratch

Select the required protocol in `main.py`:

```python
split = 'train_val_test'
```

Set the random seed in the training call:

```python
accs, kappas = trainer.train(
    lambda_center=0.005,
    lambda_triplet=0.5,
    c_ratio=0.5,
    seed=1,
    split=split
)
```

Before training, comment out the pretrained-checkpoint evaluation block and uncomment the training block at the end of `main.py`. Then run:

```bash
python main.py
```

Training logs and model checkpoints are saved under:

```text
results/<dataset>/<model_name>/<experiment_id>/
```
