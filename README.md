# Physics GNN Surrogate · Long Rollout Stabilization

[![CI](https://github.com/kohmaruworks/physics-gnn-surrogate-long-rollout/actions/workflows/ci.yml/badge.svg)](https://github.com/kohmaruworks/physics-gnn-surrogate-long-rollout/actions/workflows/ci.yml)
[![Julia](https://img.shields.io/badge/Julia-Project-9558B2?style=for-the-badge&logo=julia&logoColor=white)](https://julialang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-GNN-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch-geometric.readthedocs.io/)

大規模マルチフィジックスと自己回帰推論の安定化 — **ステップ 1–5**（Step 1: Heun + Symplectic、Step 2: Metis DDM + Halo、**Step 3: Multigrid + Tensor MP**、**Step 4: ゼロショット評価と ROI**、**Step 5: OSS 公開・CI**）。  
Julia で参照データ・グラフ IR を生成し、Python（PyG）で **Heun / Symplectic（Step 1）**、**DDM Halo（Step 2）**、**Multigrid + Tensor MP（Step 3）** を学習・合成し、**Step 4** で未知メッシュ上の自己回帰ロールアウト・エネルギー漂移・Julia 対比の高速化倍率を評価できます。

開発フロー・依存の入れ方・ライセンス表記は、姉妹プロジェクト **`physics-gnn-surrogate-basic`** および **`physics-gnn-surrogate-act`** と揃えています（リポルートの `requirements.txt`、`python3 -m venv .venv` または `uv venv`、`julia --project=.` の `Pkg.instantiate()`、成果物は `data/interim/`、MIT License）。

---

## クイックスタート（クローン → データ生成 → 学習）

以下は **Step 1（単一格子・波動）** をローカルで一通り動かす最小手順です。GPU がなくても `--cpu` で実行できます。

```bash
# 1. リポジトリ取得
git clone https://github.com/kohmaruworks/physics-gnn-surrogate-long-rollout.git
cd physics-gnn-surrogate-long-rollout

# 2. Julia：依存固定と参照軌道 JSON の生成
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. data_generation/generate_wave_data.jl

# 3. Python：仮想環境と依存
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 4. 学習（既定で data/interim/wave_rollout_step1_model.pth に保存。.gitignore 対象）
python surrogate_model/train.py --cpu   # GPU 利用時は --cpu を外す

# 5. （任意）評価パイプラインは Step 4 の README 節を参照。
#    Julia で generate_eval_data.jl を実行後、学習済みチェックポイントを指定して eval_pipeline.py を実行します。
```

CI（GitHub Actions）でも上記と同等の **Julia 生成 → インデックス変換スモーク → `train.py` 少量エポック** が自動実行されます。詳細は [.github/workflows/ci.yml](.github/workflows/ci.yml)。

---

## OSS の設計思想（コントリビューションのガイド）

第三者にも境界が伝わるよう、実装の前提を三つに絞っています。

1. **言語間の境界と役割分担**  
   **Julia**：参照物理シミュレーション・JSON IR の生成・メタデータ付きの計時。**Python（PyTorch / PyG）**：学習・推論・評価。**JSON スキーマ**で境界を固定し、再現性と差し替え容易性を優先します。
2. **インデックスの安全保障**  
   Julia は **1-based** の辺・行列インデックスをスキーマどおり JSON に書き、Python は **`convert_julia_to_python_indices`**（およびスパース COO 用変換）を**データロード経路で必ず通す**ことで、オフバイワン起因の「見えないバグ」を防ぎます。
3. **モジュール性（応用圏論的アプローチ）**  
   空間メッセージパッシング、時間積分（Heun）、Halo Exchange（DDM）、射影（Restriction / Prolongation）を **関手（Functor）として捉えうる独立コンポーネント**に分割し、**合成（composition）** でパイプラインを組み立てられるようにしています。別物理・別評価でも同じパターンを再利用しやすくします。

バグ報告・機能提案・ドキュメント修正は歓迎です。大きな変更の前に Issue で方針を相談いただけるとスムーズです。

---

## コアとなる物理法則と数式

本リポジトリが対象とする離散波動系サロゲートでは、時間更新に **Heun 法**、学習に **離散ハミルトニアンに基づくシンプレクティック制約**、評価に **自己回帰ロールアウト RMSE** を用います。

**Heun 法による時間積分**（中間状態 $\tilde{h}^{(t+1)}$ は実装で明示的に構成されます）

$$
h^{(t+1)} = h^{(t)} + \frac{\Delta t}{2} \left( f_{\theta}(h^{(t)}) + f_{\theta}(\tilde{h}^{(t+1)}) \right)
$$

**Symplectic Loss（エネルギー保存に関するペナルティ項）**

$$
\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda_{symp} \sum_{t} \left\| \mathcal{H}(h^{(t+1)}) - \mathcal{H}(h^{(t)}) \right\|^2
$$

**ゼロショット自己回帰ロールアウト誤差（評価パイプラインで使用）**  
（$\hat{u}$：予測、$u$：参照、$V$：ノード集合）

$$
\text{RMSE}_{rollout} = \sqrt{ \frac{1}{T \cdot |V|} \sum_{t=1}^{T} \sum_{i \in V} \left\| \hat{u}_i^{(t)} - u_i^{(t)} \right\|^2 }
$$

---

## プロジェクト構成

```text
physics-gnn-surrogate-long-rollout/
├── .github/workflows/ci.yml       # GitHub Actions CI（Julia データ生成 → Python スモーク）
├── data_generation/
│   ├── generate_wave_data.jl       # Step 1: 単一領域グリッド波動
│   ├── schema.json
│   ├── generate_large_wave_data.jl # Step 2: Metis DDM + halo パッチ
│   ├── schema_ddm.json
│   ├── generate_multigrid_data.jl # Step 3: fine/coarse 2:1 + R,P COO
│   ├── schema_multigrid.json
│   ├── generate_eval_data.jl      # Step 4: ゼロショット評価用 IR + Julia 計時
│   └── schema_eval.json
├── evaluation/                     # Step 4: メトリクス・プロファイラ・統合パイプライン
│   ├── metrics.py
│   ├── profiler.py
│   └── eval_pipeline.py
├── reports/                       # Step 4 レポート（既定 .gitignore、.gitkeep のみ追跡）
├── surrogate_model/
│   ├── utils/
│   │   ├── index_converter.py     # 1-based→0-based + DDM + sparse COO
│   │   └── halo_sync.py
│   ├── modules/
│   │   ├── message_passing.py
│   │   ├── integrator.py
│   │   ├── physics_loss.py
│   │   ├── ddm.py
│   │   ├── multigrid.py           # Step 3: Restriction / Prolongation
│   │   └── tensor_mp.py           # Step 3: TensorMessagePassing (einsum)
│   ├── model.py
│   ├── model_hierarchical.py      # Step 3: HierarchicalPhysicsGNN
│   ├── train.py
│   ├── train_ddm.py
│   └── train_step3.py             # Step 3 学習
├── data/interim/                # 生成 JSON・学習済み .pth（.gitignore、.gitkeep のみ追跡）
├── Project.toml                 # Julia 依存（JSON3, DifferentialEquations, Metis）
├── requirements.txt             # Python 依存（torch, torch-geometric, numpy）
├── .gitignore
├── LICENSE
└── README.md
```

### インデックス契約（関連ベースライン実装との差分）

- **本リポジトリ**: Julia はスキーマどおり **辺端点を 1-based のまま JSON に書き出し**、Python 側で **`convert_julia_to_python_indices`** をデータロード時に必ず通します（仕様で明示）。
- **`physics-gnn-surrogate-basic`**: IR 上はエクスポート時に 0-based に正規化する設計。どちらも「境界で意味を固定する」という方針は同じです。

---

## 環境構築

### Julia（データ生成）

Julia は **`Manifest.toml` に記載の `julia_version` と整合するバージョン**を推奨します（現状のロックファイルは **1.12** 系）。GitHub Actions の CI も同じマイナー系列で実行します。インストールは [juliaup](https://github.com/JuliaLang/juliaup) または公式バイナリ（補足ドキュメントが必要な場合は **`physics-gnn-surrogate-basic`** など姉妹リポジトリの `docs/` を参照）。

```bash
cd physics-gnn-surrogate-long-rollout
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. data_generation/generate_wave_data.jl
```

ルートの `Project.toml` は **`physics-gnn-surrogate-basic` と同様、「環境」用**です（`name` / `uuid` を付けない）。付けると Julia がその名前の**パッケージ**として precompile し、`src/` が無い場合に「Missing source file」で失敗します。

出力: `data/interim/wave_rollout_step1.json`（スキーマ `physics_gnn_wave_rollout_step1_v1`）。

### Python（学習）

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# または: uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt

python surrogate_model/train.py
```

主なオプション:

- `--epochs`, `--lr`, `--hidden`, `--layers`
- `--lambda-symp`: エネルギー保存ペナルティの重み \(\lambda_{\mathrm{symp}}\)
- `--rollout-min` / `--rollout-max`: エポックに応じて伸ばす **ロールアウト長（カリキュラム）**
- `--val-split`: 末尾時刻を検証に回す割合（自己回帰ロールアウト MSE を表示）

学習済み: `data/interim/wave_rollout_step1_model.pth`

---

## ステップ 2: DDM（領域分割 + Halo Exchange）

Julia で **Metis** により格子グラフを `K` 分割し、各パッチに **1-hop halo（ゴースト）** を付与した JSON（`physics_gnn_wave_rollout_ddm_v1`）を生成します。Python 側はサブドメイン単位でテンソルを保持し、**各グローバル時間ステップの Heun 後に `sync_halo_features`** で境界を同期します（`PhysicsGNNSurrogateDDM`）。`message_passing.py` / `integrator.py` 本体は変更しません。

### Julia（DDM データ）

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. data_generation/generate_large_wave_data.jl
```

出力例: `data/interim/wave_rollout_ddm_v1.json`。スキーマは `data_generation/schema_ddm.json`。

### Python（DDM 学習）

```bash
python surrogate_model/train_ddm.py
```

- **既定**: 各時間ウィンドウで **teacher-forced halo**（ゴースト行を全局 GT で上書き）によるサブドメイン損失を **`loss / K` に分解して backward** し、メモリ効率と勾配合算を両立（マイクロバッチは `--microbatch-subdomains`）。
- **`--joint-ddm-loss`**: 結合 `rollout_ddm` を 1 回の backward で最適化（厳密だがメモリ負荷大）。

学習済み: `data/interim/wave_rollout_ddm_model.pth`

---

## ステップ 3: Multigrid + Tensor メッセージパッシング（長距離相互作用）

Julia で **2:1 の構造化細／粗格子**、**全重量 Restriction `R`**（4 細頂点の平均）と **piecewise constant の Prolongation `P`** の COO を出力します（スキーマ `physics_gnn_multigrid_v1`）。Python は **`convert_julia_sparse_coo_to_torch`** で行列インデックスを一元変換し、`torch.sparse.mm` で \(h_c=R\,h_f\)、\(h_f \mathrel{+}= P\,F_{\mathrm{coarse}}(h_c)\) を合成します。局所層は **`TensorMessagePassing`**（`torch.einsum` によるボンド縮約）で、Heun / DDM とは別ファイルのため単体テスト・合成が容易です。

### Julia（Multigrid データ）

```bash
julia --project=. data_generation/generate_multigrid_data.jl
```

出力例: `data/interim/multigrid_wave_v1.json`（`schema_multigrid.json`）。

### Python（Step 3 学習）

```bash
python surrogate_model/train_step3.py
```

学習済み: `data/interim/hierarchical_step3_model.pth`

---

## ステップ 4: ゼロショット評価と ROI（Julia ↔ Python / JSON）

学習時と異なる格子・条件の **2D 波動** を Julia で解き、参照軌道と **Julia ソルバーの壁時計時間** を `schema_eval.json` 準拠の JSON に書き出します。Python はその IR を読み、`convert_julia_to_python_indices` / `convert_julia_sparse_coo_to_torch` を経由して学習済みモデルで **自己回帰ロールアウト** を行い、**累積 RMSE**・**エネルギー漂移**・**1 ステップ推論時間（CUDA Event 等）** を算出し、`julia_seconds_per_macro_step` から **Speedup（ROI）** を計算します。

### Julia（評価用データ生成 + ベースライン計時）

```bash
julia --project=. data_generation/generate_eval_data.jl
```

既定出力: `data/interim/eval_zero_shot_v1.json`。メッシュサイズや出力パスはスクリプト先頭の定数で変更できます。

### Python（評価パイプライン）

リポジトリルートから:

```bash
python evaluation/eval_pipeline.py \
  --eval-json data/interim/eval_zero_shot_v1.json \
  --checkpoint data/interim/hierarchical_step3_model.pth
```

- **`--architecture auto|heun|hierarchical`**: `auto` はチェックポイントの `meta`（例: `bond` の有無）から階層か Heun かを推定します。
- **`--cpu`**: GPU が無い環境や CPU 計測用。
- **`--max-rollout-steps`**: 0 で評価 JSON の時系列長に合わせます。
- **`--report-json`**: 既定は `reports/evaluation_results.json`。

生成されるレポートには、`rollout_rmse`、`energy_drift_max_relative`、`gnn_seconds_per_step_mean`、`speedup_vs_julia_macro_step` などが JSON で記録されます。**`reports/` 配下の生成ファイルは `.gitignore`** され、`reports/.gitkeep` のみリポジトリに含めます。

---

## ステップ 5: OSS 公開・CI

### GitHub Actions（CI）

`main` / `master` への push と Pull Request で、Ubuntu 上で次を順に実行します。

1. Julia の `Pkg.instantiate()` と **`data_generation/generate_wave_data.jl`**（Step 1 の参照 JSON 生成）
2. Python（CPU 版 PyTorch）のインストール
3. **`convert_julia_to_python_indices` を用いたインデックス変換のスモーク検証**（インラインスクリプト）
4. **`surrogate_model/train.py --epochs 2 --cpu`** によるスモーク学習

ワークフロー定義: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

---

## アルゴリズム概要

### Step 3

1. **階層 GNN**: 細格子でテンソル MP → \(h_c = R h_f\) → 粗格子でテンソル MP → \(h_f \mathrel{+}= P h_c\) → 線形デコード。
2. **Tensor MP**: エッジごとに \(m_{ij}=\sum_{\alpha\beta} A_i^\alpha W_{\alpha\beta}(e_{ij}) B_j^\beta\) を `einsum` で評価し集約。

### Step 1

1. **Heun（2 次 RK）**: 潜在状態 \(h\) に対し GNN が \(f_\theta(h)\approx dh/dt\) を与え、ドキュメント記載の更新式で \(h^{t+1}\) を計算。
2. **SymplecticLoss**: 離散ハミルトニアン \( \mathcal{H} \approx \mathrm{KE}(v) + \mathrm{PE}_{\mathrm{edges}}(u)\) を構成し、\(|\mathcal{H}^{t+1}-\mathcal{H}^t|^2\) を時間方向に集約（双方向辺を考慮して \(\lambda_{\mathrm{edges}}\) を較正）。
3. **学習**: スライディングウィンドウで **教師ありロールアウト**（予測軌道と JSON の \(u,v\) の MSE）に **`lambda_symp` 倍のシンプレクティック損失** を加算。

---

## GitHub への push について

- **`data/interim/*` は .gitignore**（巨大・再現可能な成果物のため）。サンプル JSON をリポジトリに含めたい場合は README にその旨を書き、`git add -f` 等で明示的に追加してください。
- **`articles/` と `youtube_scripts/` は .gitignore** 済みです（連載・動画向けの下書きをローカルにだけ置く想定）。リモートには含まれません。
- **`Manifest.toml`**: `julia --project=. -e 'using Pkg; Pkg.instantiate()'` で生成されます。**`physics-gnn-surrogate-basic` など姉妹環境と同様、依存固定のためリポジトリにコミットする運用**を推奨します。

---

## ライセンス

MIT License — 詳細は [LICENSE](LICENSE)。
