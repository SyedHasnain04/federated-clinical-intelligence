# Person B — Multimodal Modeling
## Implementation Plan (follow 00_SHARED_REPO_PROTOCOL.md for every phase)

Folders owned: `models/`, `data/preprocessing.py`, `tests/test_models.py`, `tests/test_preprocessing.py`.

---

### PHASE 1 — Contract Confirmation + Scaffolding

**Deliverable (exact):**
- Read Person A's `models/model_interface.py` stub (pull `main` after their Phase 1 merges). Confirm your model will implement standard `nn.Module` with `.state_dict()`/`.load_state_dict()`.
- `models/tabular.py`, `models/text.py`, `models/fusion.py` created as empty class stubs: `class TabularEncoder(nn.Module)`, `class TextEncoder(nn.Module)`, `class FusionModel(nn.Module)` — just `__init__` and `forward` signatures, no logic yet
- Decision recorded as a comment at top of `models/text.py`: frozen ClinicalBERT vs lightweight BiLSTM/TF-IDF, with one sentence why

**Test command:** `pytest tests/test_models.py::test_imports -v`
**Test must confirm:** all three classes instantiate with dummy args and `forward()` on random tensors returns a tensor of the expected shape (declare expected shape as a docstring in each class).

**Human checkpoint:** pull Person A's merged `config/base.yaml` and `client/client_app.py` yourself, read the `fit()`/`evaluate()` signature, and mentally trace how your model's `state_dict()` will flow through it. If it doesn't fit, comment on their PR/Slack now, not later.

---

### PHASE 2 — Preprocessing Pipeline

**Deliverable (exact):**
- `data/preprocessing.py` with `def preprocess_tabular(df: pd.DataFrame, fit_stats: dict = None) -> (pd.DataFrame, dict)` — returns cleaned df + the fit stats (mean/std/mode used), so stats can be computed per-node, not globally
- `def preprocess_text(notes: list[str]) -> list[str]` — cleaning/normalization only, no tokenization yet
- Must run once per `data/nodes/node_{i}/data.parquet` (Person A's Phase 2 output) — write `data/nodes/node_{i}/data_clean.parquet` for each node

**Test command:** `pytest tests/test_preprocessing.py -v`
**Tests must confirm (exact assertions):**
1. `assert df_clean.isnull().sum().sum() == 0`
2. run `preprocess_tabular` independently on node 0 and node 1 data, `assert fit_stats["node0"] != fit_stats["node1"]` (proves no global leakage)
3. text cleaning removes non-printable characters and strips leading/trailing whitespace on a sample with injected noise

**Fix loop:** if assertion 2 fails (stats identical across nodes), you have a global-leakage bug — do not proceed until fixed.

**Human checkpoint:** print 5 raw vs cleaned records (mixed tabular+text) and read them yourself before merging.

---

### PHASE 3 — Tabular Branch

**Deliverable (exact):**
- `models/tabular.py`: `class TabularEncoder(nn.Module)` — MLP, `__init__(self, input_dim, hidden_dims=[64,32], output_dim=16, mode="embedding")` where `mode` is `"embedding"` (returns 16-dim vector) or `"logit"` (returns 1 sigmoid logit)
- `models/train_centralized_tabular.py` — standalone script: loads all nodes' cleaned data combined, trains `TabularEncoder(mode="logit")` centrally, reports accuracy/precision/recall/F1, saves to `models/baselines/tabular_centralized_metrics.json`

**Test command:** `pytest tests/test_models.py::test_tabular -v`
**Tests must confirm:** forward pass shape correct in both modes; 5-epoch training run on synthetic data produces non-NaN loss that decreases; save/load round-trip produces identical predictions on a fixed input.

**Human checkpoint:** open `models/baselines/tabular_centralized_metrics.json`, sanity-check the accuracy number is plausible (not 50% = random, not 100% = leakage) before treating it as your baseline reference.

---

### PHASE 4 — Text Branch

**Deliverable (exact):**
- `models/text.py`: `class TextEncoder(nn.Module)` — if frozen ClinicalBERT: wraps `AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")`, frozen weights, `forward(input_ids, attention_mask) -> 768-dim pooled embedding`, plus `models/embedding_cache.py` with `def get_or_compute_embedding(note_id, text) -> np.ndarray` caching to `data/nodes/node_{i}/text_embeddings.npz`
- If lightweight fallback: TF-IDF (max_features=5000) + 2-layer MLP down to 64-dim

**Test command:** `pytest tests/test_models.py::test_text -v`
**Tests must confirm:** embedding output shape/dtype correct; calling `get_or_compute_embedding` twice on the same note_id returns identical arrays (cache hit, not recompute); a cosine-similarity script comparing 3 similar-topic notes vs 3 unrelated notes shows similar-pair average similarity > unrelated-pair average similarity.

**Fix loop:** if the cosine-similarity sanity check fails (similar notes aren't more similar than unrelated ones), check tokenizer/model mismatch before proceeding — this is a silent-bug pattern.

**Human checkpoint:** personally read the 3 similar / 3 unrelated note pairs and their similarity scores before merging.

---

### PHASE 5 — Fusion + Joint Integration (JOINT with Person A)

**Deliverable (exact):**
- `models/fusion.py`: `class FusionModel(nn.Module)` — `__init__(self, tabular_dim=16, text_dim=64_or_768, hidden_dim=32)`, concatenates `TabularEncoder(mode="embedding")` output + `TextEncoder` output, MLP head to 1 sigmoid logit
- `models/train_centralized_ablation.py` — trains and reports tabular-only, text-only, fusion, saves `models/baselines/ablation_results.json`
- Coordinate live with Person A: your `FusionModel` gets imported into their `server/run_simulation.py` in this same phase

**Test command:** `pytest tests/test_models.py::test_fusion -v` AND (jointly) Person A's `test_real_model_round`
**Test must confirm:** fusion forward pass shape correct; ablation script runs all 3 configs without error and produces a results table with 3 non-degenerate accuracy numbers.

**Fix loop:** any interface error when Person A wires in your model — fix on your side to match the Phase 1 contract exactly, don't ask them to work around your model's quirks.

**Human checkpoint (joint):** watch one real federated round complete with your fusion model live with Person A before either of you merges. Separately, review the ablation table — if fusion underperforms tabular-only, note this honestly rather than adjusting hyperparameters until it "looks right."

---

### PHASE 6 — Robustness, Docs, Final Checkpoint

**Deliverable (exact):**
- `models/train_centralized_ablation.py` extended: re-run ablation with Person A's DP wrapper on/off (import from `privacy/dp_wrapper.py`), append to `ablation_results.json`
- `models/inference.py`: `def predict(tabular_input: dict, note_text: str, checkpoint_path: str) -> dict` returning `{"prediction": float, "confidence": float, "top_features": list, "top_tokens": list}` (simple SHAP for tabular, attention-weight-based for text)
- `models/README.md`: architecture, ablation table, checkpoint loading instructions
- Export final best checkpoint to `models/checkpoints/final_fusion_model.pt`

**Test command:** `pytest tests/test_models.py -v` (full file)
**Tests must confirm:** `inference.py`'s `predict()` runs end-to-end on 5 hardcoded example inputs without error and returns correctly-typed output dict for each.

**Human checkpoint:** personally run the 5 example predictions and read the outputs — confirm they're sane (e.g. clearly-severe example predicts high risk) before handing `final_fusion_model.pt` to Person C for the dashboard.
