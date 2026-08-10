# Person B — Prompt Guide (use with code agent, in order, per phase)

Before every phase: `git checkout main && git pull origin main`, then branch per 00_SHARED_REPO_PROTOCOL.md.

---

## Phase 1

**Prompt 1 (build):**
> Pull the latest main and show me the contents of `models/model_interface.py` and `client/client_app.py` written by a teammate. Confirm whether a standard PyTorch `nn.Module` with `.state_dict()`/`.load_state_dict()` fits their expected format — flag explicitly if it doesn't. Then create `models/tabular.py`, `models/text.py`, `models/fusion.py` as stub classes: `class TabularEncoder(nn.Module)`, `class TextEncoder(nn.Module)`, `class FusionModel(nn.Module)`, each with `__init__` and `forward` only (no real logic yet), and a docstring stating expected input/output shape. Add a one-line comment at the top of `models/text.py` recording the decision: frozen ClinicalBERT vs lightweight TF-IDF/BiLSTM, and why, given [state your compute budget].

**Prompt 2 (test + fix loop):**
> Write `tests/test_models.py::test_imports` that instantiates all three stub classes with dummy dimensions and asserts `forward()` on random tensors returns tensors of the documented shape. Run `pytest tests/test_models.py::test_imports -v`. Fix any failures and re-run until green.

**Prompt 3 (push):**
> Commit as "phase1: model stubs + contract review", push to `feature/personB-phase1-stubs`, open PR noting any contract concerns for Person A to review.

---

## Phase 2

**Prompt 1 (build):**
> Write `data/preprocessing.py` with `def preprocess_tabular(df, fit_stats=None) -> (df_clean, fit_stats)` (impute/normalize, computing stats fresh if `fit_stats` is None — never use global stats across nodes) and `def preprocess_text(notes: list[str]) -> list[str]` (clean whitespace/non-printable chars). Write a script that runs this over each `data/nodes/node_{i}/data.parquet` from Person A's partitioning output and writes `data/nodes/node_{i}/data_clean.parquet`.

**Prompt 2 (test + fix loop):**
> Write `tests/test_preprocessing.py` asserting: (1) zero nulls remain post-cleaning, (2) running `preprocess_tabular` separately on node 0 vs node 1 data produces different `fit_stats` (proves no cross-node leakage), (3) text cleaning strips injected noise (non-printable chars, extra whitespace) from a synthetic test string. Run `pytest tests/test_preprocessing.py -v`. If test 2 fails (stats identical across nodes), this is a leakage bug — fix and re-run until it passes, do not skip this.

**Prompt 3 (push):**
> Print 5 sample before/after records (mixed tabular + text) so I can review them directly in your response before I commit. Then commit as "phase2: preprocessing pipeline", push to `feature/personB-phase2-preprocessing`, open PR with the sample output pasted in.

---

## Phase 3

**Prompt 1 (build):**
> Implement `models/tabular.py`'s `TabularEncoder(nn.Module)`: `__init__(self, input_dim, hidden_dims=[64,32], output_dim=16, mode="embedding")`, MLP, `mode="embedding"` returns a 16-dim vector, `mode="logit"` returns a single sigmoid logit. Write `models/train_centralized_tabular.py`: loads all nodes' cleaned data combined, trains `TabularEncoder(mode="logit")` centrally with a standard train/val split, reports accuracy/precision/recall/F1, saves results to `models/baselines/tabular_centralized_metrics.json`.

**Prompt 2 (test + fix loop):**
> Write tests for both modes' output shapes, confirm a 5-epoch training run on synthetic data has non-NaN, decreasing loss, and confirm save/load round-trip gives identical predictions on a fixed input. Run them, fix failures, re-run until green. Then actually run `train_centralized_tabular.py` and show me the full metrics output.

**Prompt 3 (push):**
> Commit as "phase3: tabular branch + centralized baseline", push to `feature/personB-phase3-tabular`, open PR with the baseline metrics in the description.

---

## Phase 4

**Prompt 1 (build):**
> Implement `models/text.py`'s `TextEncoder(nn.Module)` using [frozen Bio_ClinicalBERT / TF-IDF+MLP — per Phase 1 decision]. If ClinicalBERT: freeze all weights, `forward(input_ids, attention_mask)` returns 768-dim pooled embedding. Write `models/embedding_cache.py` with `get_or_compute_embedding(note_id, text)` caching to `data/nodes/node_{i}/text_embeddings.npz` so repeated calls don't recompute.

**Prompt 2 (test + fix loop):**
> Write tests: output shape/dtype correctness, cache-hit returns identical array on repeated calls for the same note_id. Also write a small script computing cosine similarity between 3 pairs of similar-topic clinical notes and 3 pairs of clearly unrelated notes, and assert the similar-pair average similarity is higher than the unrelated-pair average. Run everything, fix failures, and show me the actual similarity numbers and the note text pairs used, so I can read them myself.

**Prompt 3 (push):**
> Commit as "phase4: text branch + embedding cache", push to `feature/personB-phase4-text`, open PR with the cosine-similarity sanity check output included.

---

## Phase 5 (coordinate live with Person A before running Prompt 3)

**Prompt 1 (build):**
> Implement `models/fusion.py`'s `FusionModel(nn.Module)`: `__init__(self, tabular_dim=16, text_dim=[64 or 768], hidden_dim=32)`, concatenates `TabularEncoder(mode="embedding")` and `TextEncoder` outputs, MLP head to 1 sigmoid logit. Write `models/train_centralized_ablation.py` training and comparing tabular-only, text-only, and fusion configs, saving results to `models/baselines/ablation_results.json`.

**Prompt 2 (test + fix loop):**
> Write a shape/type test for `FusionModel.forward()`. Run the ablation script and show me the full 3-way results table. If any config trains to a degenerate result (NaN, near-random for all 3), diagnose and fix before proceeding — but if fusion genuinely underperforms tabular-only, do NOT tune it until it "wins"; report the honest result.

**Prompt 3 (push, after live session with Person A):**
> [Do this together with Person A live] Import `FusionModel` into `server/run_simulation.py` and run one full federated round together — fix any interface mismatches on my side to match the agreed contract exactly. Once it runs clean, commit as "phase5: fusion model + federated integration", push to `feature/personB-phase5-fusion`, open PR noting it needs Person A's sign-off, and paste the ablation table in the description.

---

## Phase 6

**Prompt 1 (build):**
> Extend `models/train_centralized_ablation.py` to also run each config with Person A's DP wrapper (`privacy/dp_wrapper.py`) on and off, appending results to `ablation_results.json`. Write `models/inference.py`'s `predict(tabular_input: dict, note_text: str, checkpoint_path: str) -> dict` returning prediction, confidence, top contributing tabular features (simple SHAP), and top contributing text tokens (attention-weight based). Export the best-performing checkpoint to `models/checkpoints/final_fusion_model.pt`.

**Prompt 2 (test + fix loop):**
> Write 5 hardcoded example inputs (mix of clearly-low-risk and clearly-high-risk profiles) and run `predict()` on each. Show me the full output for all 5 so I can manually judge whether the predictions look sane before trusting the checkpoint for the demo. Fix and re-run if any look wrong or error out.

**Prompt 3 (push):**
> Write `models/README.md` covering architecture, final ablation table, and checkpoint loading instructions. Run the full `tests/test_models.py` suite one more time. Commit as "phase6: inference + docs + final checkpoint", push to `feature/personB-phase6-final`, open PR, and after I approve, merge to main.
