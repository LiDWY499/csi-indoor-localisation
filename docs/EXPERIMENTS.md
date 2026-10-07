# CSI Localisation Project

This repository contains the code used for the CSI indoor localisation experiments. It supports Qiang Lab/Meeting Room experiments, classical baselines, DeepFi-style models, CNN, TCN, tuned Transformer models, and the Figshare Intel 5300 orientation-robustness benchmark.


## Repository structure

```text
Li_Wang_Yau_Derek_UG_Project_SuppMat/
├── README.md
├── requirements.txt
└── src/
    ├── convert_qiang_dataset.py
    ├── figshare_intel5300_prepare.py
    ├── deepfi_pipeline.py
    ├── classical_csi_baselines_windowed.py
    ├── cnn_csi_windowed.py
    ├── tcn_csi_windowed.py
    ├── transformer_csi_windowed.py
    └── transformer_csi_tuned.py
```


## 1. Setup

Open Windows Command Prompt in the project folder.

```bat
cd path\to\Li_Wang_Yau_Derek_UG_Project_SuppMat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For an NVIDIA GPU, install a CUDA-enabled PyTorch build from the official PyTorch selector if `torch.cuda.is_available()` returns `False`.

```bat
python -c "import torch; print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## 2. Download datasets

### 2.1 Qiang CSI dataset

Download the Qiang CSI dataset repository from GitHub:

```text
https://github.com/qiang5love1314/CSI-dataset
```

Place it under:

```text
path\to\Li_Wang_Yau_Derek_UG_Project_SuppMat\CSI-dataset
```

The Qiang dataset contains the Lab and Meeting Room scenarios. The raw files are named like `coordinate715.mat`, where the coordinate code is converted into x/y position by `convert_qiang_dataset.py`.

### 2.2 Figshare Intel 5300 dataset

Download the Figshare dataset:

```text
https://figshare.com/articles/dataset/WiFi_CSI_dataset_for_device-to-device_localization/20943706
```

Extract the 10 `.mat` files into:

```text
path\to\Li_Wang_Yau_Derek_UG_Project_SuppMat\figshare_intel5300\wifi_csi_data_loc
```

Expected filenames include:

```text
loc_30deg_1m.mat
loc_30deg_2m.mat
...
loc_minus60deg_5m.mat
```

## 3. Prepare Qiang Lab and Meeting Room data

Convert the downloaded Qiang layout into one folder per reference point.

```bat
python src\convert_qiang_dataset.py --src "CSI-dataset\Lab Dataset" --dst data\qiang_lab_converted --copy
python src\convert_qiang_dataset.py --src "CSI-dataset\Meeting Room Dataset" --dst data\qiang_meeting_converted --copy
```

Prepare processed `.npz` files.

```bat
python src\deepfi_pipeline.py prepare --raw-root data\qiang_lab_converted --out-dir runs\lab_prepared --train-ratio 0.8
python src\deepfi_pipeline.py prepare --raw-root data\qiang_meeting_converted --out-dir runs\meeting_prepared --train-ratio 0.8
```

Check processed shapes.

```bat
python -c "import numpy as np; d=np.load(r'runs\lab_prepared\processed_dataset.npz'); print(d['X_train'].shape, d['X_test'].shape, len(set(d['rp_train'].tolist())))"
python -c "import numpy as np; d=np.load(r'runs\meeting_prepared\processed_dataset.npz'); print(d['X_train'].shape, d['X_test'].shape, len(set(d['rp_train'].tolist())))"
```

## 4. Run classical baselines

Lab:

```bat
python src\classical_csi_baselines_windowed.py --processed runs\lab_prepared\processed_dataset.npz --out-dir runs\classical_lab_window10_all_argmax --window-size 10 --aggregation argmax
```

Meeting Room:

```bat
python src\classical_csi_baselines_windowed.py --processed runs\meeting_prepared\processed_dataset.npz --out-dir runs\classical_meeting_window10_all_weighted --window-size 10 --aggregation weighted
```

## 5. Run DeepFi-style model

Lab:

```bat
python src\deepfi_pipeline.py train --processed runs\lab_prepared\processed_dataset.npz --out-dir runs\lab_models --epochs 40 --batch-size 64 --lr 0.001
python src\deepfi_pipeline.py evaluate --processed runs\lab_prepared\processed_dataset.npz --model-dir runs\lab_models --out-dir runs\lab_eval --window-size 10
```

Meeting Room:

```bat
python src\deepfi_pipeline.py train --processed runs\meeting_prepared\processed_dataset.npz --out-dir runs\meeting_models --epochs 40 --batch-size 64 --lr 0.001
python src\deepfi_pipeline.py evaluate --processed runs\meeting_prepared\processed_dataset.npz --model-dir runs\meeting_models --out-dir runs\meeting_eval --window-size 10
```

## 6. Run CNN

Lab:

```bat
python src\cnn_csi_windowed.py train --processed runs\lab_prepared\processed_dataset.npz --out-dir runs\cnn_lab --window-size 10 --channels 32 --dropout 0.2 --epochs 30 --batch-size 64 --lr 0.001
python src\cnn_csi_windowed.py evaluate --processed runs\lab_prepared\processed_dataset.npz --model runs\cnn_lab\models\best_cnn.pt --out-dir runs\cnn_lab_eval_argmax --aggregation argmax
```

Meeting Room:

```bat
python src\cnn_csi_windowed.py train --processed runs\meeting_prepared\processed_dataset.npz --out-dir runs\cnn_meeting --window-size 10 --channels 32 --dropout 0.2 --epochs 30 --batch-size 64 --lr 0.001
python src\cnn_csi_windowed.py evaluate --processed runs\meeting_prepared\processed_dataset.npz --model runs\cnn_meeting\models\best_cnn.pt --out-dir runs\cnn_meeting_eval_argmax --aggregation argmax
```

## 7. Run TCN

Lab:

```bat
python src\tcn_csi_windowed.py train --processed runs\lab_prepared\processed_dataset.npz --out-dir runs\tcn_lab --window-size 10 --proj-dim 64 --channels 64 --levels 4 --kernel-size 3 --dropout 0.2 --epochs 30 --batch-size 64 --lr 0.001
python src\tcn_csi_windowed.py evaluate --processed runs\lab_prepared\processed_dataset.npz --model runs\tcn_lab\models\best_tcn.pt --out-dir runs\tcn_lab_eval_argmax --aggregation argmax
```

Meeting Room:

```bat
python src\tcn_csi_windowed.py train --processed runs\meeting_prepared\processed_dataset.npz --out-dir runs\tcn_meeting --window-size 10 --proj-dim 64 --channels 64 --levels 4 --kernel-size 3 --dropout 0.2 --epochs 30 --batch-size 64 --lr 0.001
python src\tcn_csi_windowed.py evaluate --processed runs\meeting_prepared\processed_dataset.npz --model runs\tcn_meeting\models\best_tcn.pt --out-dir runs\tcn_meeting_eval_argmax --aggregation argmax
```

## 8. Run tuned Transformer

Final Lab model:

```bat
python src\transformer_csi_tuned.py train --processed runs\lab_prepared\processed_dataset.npz --out-dir runs\transformer_lab_focus_C --window-size 10 --d-model 128 --nhead 4 --num-layers 2 --dim-feedforward 384 --dropout 0.2 --input-dropout 0.05 --label-smoothing 0.02 --weight-decay 0.0001 --epochs 80 --batch-size 128 --lr 0.0005 --patience 15
python src\transformer_csi_tuned.py evaluate --processed runs\lab_prepared\processed_dataset.npz --model runs\transformer_lab_focus_C\models\best_transformer.pt --out-dir runs\transformer_lab_focus_C_eval_argmax --aggregation argmax
```

Final Meeting Room model:

```bat
python src\transformer_csi_tuned.py train --processed runs\meeting_prepared\processed_dataset.npz --out-dir runs\transformer_meeting_focus_C --window-size 10 --d-model 128 --nhead 4 --num-layers 2 --dim-feedforward 384 --dropout 0.2 --input-dropout 0.05 --label-smoothing 0.02 --weight-decay 0.0001 --epochs 80 --batch-size 128 --lr 0.0005 --patience 15
python src\transformer_csi_tuned.py evaluate --processed runs\meeting_prepared\processed_dataset.npz --model runs\transformer_meeting_focus_C\models\best_transformer.pt --out-dir runs\transformer_meeting_focus_C_eval_argmax --aggregation argmax
```

Optional tuning search:

```bat
python src\transformer_csi_tuned.py tune --processed runs\lab_prepared\processed_dataset.npz --out-dir runs\tune_transformer_lab_overnight --preset overnight --window-size 10 --epochs 80 --batch-size 128 --patience 15
python src\transformer_csi_tuned.py tune --processed runs\meeting_prepared\processed_dataset.npz --out-dir runs\tune_transformer_meeting_overnight --preset overnight --window-size 10 --epochs 80 --batch-size 128 --patience 15
```

## 9. Prepare Figshare robustness datasets

Mixed orientation benchmark:

```bat
python src\figshare_intel5300_prepare.py --data-dir figshare_intel5300\wifi_csi_data_loc --out-dir runs\fig_mixed_coordinate_50k --mode mixed-coordinate --max-packets-per-condition 50000
```

Orientation transfer, 30 degrees to minus 60 degrees:

```bat
python src\figshare_intel5300_prepare.py --data-dir figshare_intel5300\wifi_csi_data_loc --out-dir runs\fig_train30_testminus60_50k --mode train30-testminus60 --max-packets-per-condition 50000
```

Orientation transfer, minus 60 degrees to 30 degrees:

```bat
python src\figshare_intel5300_prepare.py --data-dir figshare_intel5300\wifi_csi_data_loc --out-dir runs\fig_trainminus60_test30_50k --mode trainminus60-test30 --max-packets-per-condition 50000
```

## 10. Run Figshare experiments

Classical baselines:

```bat
python src\classical_csi_baselines_windowed.py --processed runs\fig_mixed_coordinate_50k\processed_dataset.npz --out-dir runs\classical_fig_mixed_coordinate_50k_argmax --window-size 10 --aggregation argmax
python src\classical_csi_baselines_windowed.py --processed runs\fig_train30_testminus60_50k\processed_dataset.npz --out-dir runs\classical_fig_train30_testminus60_50k_argmax --window-size 10 --aggregation argmax
python src\classical_csi_baselines_windowed.py --processed runs\fig_trainminus60_test30_50k\processed_dataset.npz --out-dir runs\classical_fig_trainminus60_test30_50k_argmax --window-size 10 --aggregation argmax
```

Transformer:

```bat
python src\transformer_csi_tuned.py train --processed runs\fig_mixed_coordinate_50k\processed_dataset.npz --out-dir runs\transformer_fig_mixed_coordinate_50k --window-size 10 --d-model 128 --nhead 4 --num-layers 2 --dim-feedforward 384 --dropout 0.2 --input-dropout 0.05 --label-smoothing 0.02 --weight-decay 0.0001 --epochs 40 --batch-size 256 --lr 0.0005 --patience 8
python src\transformer_csi_tuned.py evaluate --processed runs\fig_mixed_coordinate_50k\processed_dataset.npz --model runs\transformer_fig_mixed_coordinate_50k\models\best_transformer.pt --out-dir runs\transformer_fig_mixed_coordinate_50k_eval_argmax --aggregation argmax

python src\transformer_csi_tuned.py train --processed runs\fig_train30_testminus60_50k\processed_dataset.npz --out-dir runs\transformer_fig_train30_testminus60_50k --window-size 10 --d-model 128 --nhead 4 --num-layers 2 --dim-feedforward 384 --dropout 0.2 --input-dropout 0.05 --label-smoothing 0.02 --weight-decay 0.0001 --epochs 40 --batch-size 256 --lr 0.0005 --patience 8
python src\transformer_csi_tuned.py evaluate --processed runs\fig_train30_testminus60_50k\processed_dataset.npz --model runs\transformer_fig_train30_testminus60_50k\models\best_transformer.pt --out-dir runs\transformer_fig_train30_testminus60_50k_eval_argmax --aggregation argmax

python src\transformer_csi_tuned.py train --processed runs\fig_trainminus60_test30_50k\processed_dataset.npz --out-dir runs\transformer_fig_trainminus60_test30_50k --window-size 10 --d-model 128 --nhead 4 --num-layers 2 --dim-feedforward 384 --dropout 0.2 --input-dropout 0.05 --label-smoothing 0.02 --weight-decay 0.0001 --epochs 40 --batch-size 256 --lr 0.0005 --patience 8
python src\transformer_csi_tuned.py evaluate --processed runs\fig_trainminus60_test30_50k\processed_dataset.npz --model runs\transformer_fig_trainminus60_test30_50k\models\best_transformer.pt --out-dir runs\transformer_fig_trainminus60_test30_50k_eval_argmax --aggregation argmax
```

## 11. Check results

Print all metric files:

```bat
dir runs\metrics.json /s /b
dir runs\metrics_per_model.json /s /b
dir runs\metrics_summary.json /s /b
```

Save all metrics into one text file:

```bat
python -c "from pathlib import Path; paths=sorted(list(Path('runs').rglob('metrics.json'))+list(Path('runs').rglob('metrics_per_model.json'))+list(Path('runs').rglob('metrics_summary.json'))); Path('ALL_REPORT_METRICS_DUMP.txt').write_text(''.join('\n\n===== '+str(p)+' =====\n'+p.read_text(errors='ignore') for p in paths), encoding='utf-8'); print('saved', len(paths), 'metric files')"
notepad ALL_REPORT_METRICS_DUMP.txt
```

## 12. Expected outputs

Each experiment writes outputs into `runs\...`, including:

```text
metrics.json
metrics_per_model.json
metrics_summary.json
predictions.csv
predictions_<model>.csv
models\best_*.pt
train_history.json
tune_results_ranked.csv
```


