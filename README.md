# TE-Q-Transformer: Hybrid Quantum-Classical Transformer for Li-ion Battery SOH Estimation

Reproducible artifacts (notebooks, trained checkpoints, logs, and figures) for **TE-Q-Transformer**, a hybrid quantum-classical model for Li-ion battery State-of-Health (SOH) prediction. The model injects an Arrhenius-governed temperature embedding into a 4-qubit PennyLane quantum circuit, smooths the resulting sequence with a Conv1D temporal smoother, and feeds a CLS-tokenized, Pre-LN transformer encoder to regress SOH.

This repository accompanies a manuscript prepared for submission to an Elsevier journal (Computers and Electrical Engineering — Q1). Citation details will be added once a DOI is available (see [Citation](#citation)).

---

## Repository contents

All notebooks and output archives are at the repo root. Output archives are zipped as uploaded from Kaggle; unzip them locally to inspect checkpoints, logs, and figures (see [Unpacking the output archives](#unpacking-the-output-archives)).

| File | Description |
|---|---|
| `Nasa_TE-Q-Transformer.ipynb` | Full TE-Q-Transformer pipeline: data protocol, architecture, training, evaluation, export. Run on Kaggle T4 GPU. This is the main model notebook. |
| `nasa-te-q-transformer-gru.ipynb` | GRU baseline notebook (same data protocol and evaluation harness as the main model). |
| `nasa-te-q-transformer-lstm.ipynb` | LSTM baseline notebook. |
| `nasa-te-q-transformer-transformer.ipynb` | Classical Transformer baseline notebook. |
| `nasa-te-q-transformer-qlstm.ipynb` | QLSTM (quantum LSTM) baseline notebook. |
| `nasa-te-q-transformer-qnn-gru.ipynb` | QNN-GRU (quantum GRU) baseline notebook. |
| `NASA_TEQ_Transformer_ablationOutputs.zip` | Outputs of the TE-Q-Transformer's `01_rich_entangler` run — checkpoints, training history, config, per-cell metrics, figures. **This is the canonical TE-Q-Transformer result reported in the paper** (see note below). |
| `NASA_GRU_Baseline_Outputs.zip` | GRU baseline outputs: checkpoints, config, training history, per-cell metrics, figures. |
| `NASA_lstm_baseline_Outputs.zip` | LSTM baseline outputs. |
| `NASA_Transformer_Baseline_Outputs.zip` | Classical Transformer baseline outputs. |
| `NASA_qlstm_baseline_Outputs.zip` | QLSTM baseline outputs. |
| `NASA_qnn_gru_baseline_Outputs.zip` | QNN-GRU baseline outputs. |
| `LICENSE` | MIT license. |
| `README.md` | This file. |

> **Note on the ablation zip:** `NASA_TEQ_Transformer_ablationOutputs.zip` contains one folder, `01_rich_entangler/`, which is both the TE-Q-Transformer's best ablation variant *and* the canonical model reported throughout the paper — there is no separate "main model" archive. Treat `01_rich_entangler/` inside that zip as the headline TE-Q-Transformer result when comparing against the five baseline zips.

An ablation-sweep notebook (`nasa_teq_component_ablation_study.ipynb`, which produced this zip) is not yet in this repo — add it if you want the full sweep code alongside its output archive.

---

## Model summary

- **Task:** Regress SOH (State of Health) for Li-ion cells from voltage, current, time, and temperature sequences.
- **Quantum layer:** 4-qubit circuit (`default.qubit`, PennyLane), Arrhenius-based temperature encoding (SEI growth + lithium-plating terms) injected via an `RY` rotation, followed by a "rich" entangling layer.
- **Classical backbone:** Conv1D temporal smoother → CLS token → Pre-LN Transformer encoder (3 layers, 2 heads, `d_model=64`, feedforward=64) → scalar SOH regression head.
- **Sequence length:** 512. **Input features:** 4 (Voltage, Current, Time_norm, Temperature_C — temperature kept raw/unscaled for the Arrhenius embedding).
- **Validation protocol:** Hybrid multi-temperature, leave-cell-out cross-validation. Train cells: `B0005, B0006, B0007, B0029, B0030, B0031`. Test: unseen cells `B0018`, `B0032`, plus the chronologically held-out final 30% of `B0053` (`B0053_test`).
- **Datasets:** NASA Prognostics Center of Excellence Battery Data Set (Li-ion cells `B0005`–`B0053`). **Not redistributed here** — see [Data](#data) below.

## Results

Macro metrics across the three held-out test segments (`B0018`, `B0032`, `B0053_test`):

| Model | Macro RMSE ↓ | Macro MAE ↓ | Macro R² ↑ |
|---|---|---|---|
| **TE-Q-Transformer (rich entangler)** | **0.01678** | **0.01409** | **0.8699** |
| QNN-GRU | 0.03392 | 0.02882 | -0.5235 |
| LSTM | 0.04086 | 0.03549 | 0.0010 |
| QLSTM | 0.04158 | 0.03684 | -0.1778 |
| Transformer | 0.04748 | 0.04110 | -0.3277 |
| GRU | 0.04759 | 0.04121 | -0.6906 |

TE-Q-Transformer reduces macro RMSE by roughly 51–65% relative to all five baselines and is the only model with a strongly positive R² across all three held-out segments. Full per-cell breakdowns (RMSE, MAE, MAPE, R², MaxE) are inside each zip's `*_result.json` / `*_summary.json`.

---

## Setup

```bash
git clone https://github.com/LisanHub/teq-transformer-battery-soh.git
cd teq-transformer-battery-soh

python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch pennylane numpy pandas scikit-learn matplotlib seaborn jupyter
```
Pin exact versions from your Kaggle environment with `pip freeze | grep -iE "torch|pennylane|numpy|pandas|scikit|matplotlib|seaborn"` if you want a fully locked environment.

## Unpacking the output archives

Each zip mirrors the folder Kaggle wrote during training. Unzip whichever one you need:

```bash
unzip NASA_TEQ_Transformer_ablationOutputs.zip
# → nasa_results/ablation_study/01_rich_entangler/
#     nasa_teq_transformer_best.pth
#     nasa_teq_transformer_last.pth
#     nasa_training_config.json
#     nasa_training_history.json
#     ablation_result.json
#     figures/

unzip NASA_GRU_Baseline_Outputs.zip
# → gru_results/gru_baseline/01_baseline/
#     gru_best.pth, gru_last.pth
#     gru_training_config.json, gru_training_history.json
#     gru_result.json
#     gru_vs_teq_comparison.{json,csv}
#     figures/
```
The other three baseline zips (`lstm`, `transformer`, `qlstm`, `qnn_gru`) follow the identical `*_results/*_baseline/01_baseline/` layout with model-name-prefixed files.

## Data

The notebooks expect the **NASA Li-ion Battery Data Set** (cells `B0005, B0006, B0007, B0018, B0029, B0030, B0031, B0032, B0053`). This dataset is **not included** in this repository due to size and license terms.

1. Download from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) (or the Kaggle mirror used in the original notebooks).
2. Place the raw `.mat`/`.csv` files under a local `data/` directory.
3. Update the dataset path constant in the first cell of each notebook (set during the "Kaggle Setup" cell) to point at your local `data/` path.

## Reproducing the results

Run in this order:

1. **`Nasa_TE-Q-Transformer.ipynb`** — trains and evaluates TE-Q-Transformer end to end; exports checkpoints, training history, and per-cell metrics (the contents mirrored in `NASA_TEQ_Transformer_ablationOutputs.zip`). This notebook also contains the ablation-variant scaffolding.
2. **`nasa-te-q-transformer-{gru,lstm,transformer,qlstm,qnn-gru}.ipynb`** — each baseline notebook uses the identical data protocol and evaluation harness, and can be run independently in any order. Each exports a `*_vs_teq_comparison.{json,csv}` for direct comparison against the TE-Q-Transformer run.

All notebooks were authored for Kaggle (T4 GPU). To run locally with a GPU, drop the `!pip -q install ...` Kaggle setup cell in favor of your local environment, and confirm `torch.cuda.is_available()` returns `True` — training will fall back to CPU otherwise (much slower, but numerically reproducible).

**Reproducibility notes:**
- Best-epoch selection uses early stopping (`patience=20`, monitored on validation macro RMSE).
- Optimizer: Adam, `lr=1e-3`, `weight_decay=5e-2`, `ReduceLROnPlateau` (`factor=0.5`, `patience=10`), gradient clipping at `1.0` norm.
- Batch size 8, up to 80 epochs.
- Seeds are fixed in each notebook's setup cell; exact bitwise reproducibility across hardware (CPU vs. different GPU models) is not guaranteed for the quantum circuit simulation, though metric-level reproducibility (RMSE/MAE/R² within a small tolerance) should hold.

## Loading a trained checkpoint

```python
import torch
from pathlib import Path

# after unzipping NASA_TEQ_Transformer_ablationOutputs.zip
ckpt_path = Path("nasa_results/ablation_study/01_rich_entangler/nasa_teq_transformer_best.pth")
state_dict = torch.load(ckpt_path, map_location="cpu")

model = TEQTransformer(**config)   # instantiate using nasa_training_config.json from the same folder
model.load_state_dict(state_dict)
model.eval()
```
See `Nasa_TE-Q-Transformer.ipynb` for the full `TEQTransformer` class definition and config dataclass used to instantiate the model before loading weights.

## License

MIT — see `LICENSE`. (Adjust if your institution or the funding grant imposes different IP terms; confirm with Kabir's Research Group / DIU before publishing.)

## Citation

```bibtex
@article{lisan_teq_transformer,
  title   = {TE-Q-Transformer: A Hybrid Quantum-Classical Transformer for Li-ion Battery State-of-Health Estimation},
  author  = {Al Lisan, Zadid, Sifat, Md Sulyman Islam, and Kabir, Md Alamgir},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```
Update with the final author list, volume/issue, and DOI once accepted. See also the related published systematic review:
> Al Lisan, Z. et al. "Quantum machine learning in predictive maintenance: A comprehensive state-of-the-art review." *Computers and Electrical Engineering*, Vol. 139 (2026), Article 111374.

## Acknowledgments

Research conducted under Kabir's Research Group (Supervisor: Dr. Md Alamgir Kabir (kabir.cse@diu.edu.bd)), Department of CSE, Daffodil International University, Dhaka, Bangladesh.
