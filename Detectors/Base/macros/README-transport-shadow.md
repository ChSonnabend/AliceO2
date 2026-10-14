# ONNX transport scoring without rejecting tracks

Branch: `feature/onnx-primary-pruning-keep-with-threshold`, based on
`feature/onnx-primary-pruning` at d45ac8236c on Hydra.

## Runtime configuration

The ONNX input is the existing `simnet.birth.v1` 34-feature contract. Use a
**deployment** graph, not the old 19/21-input training export. Set:

```
Stack.transportPrimary=onnx
Stack.transportPrimaryOnnxModelFile=/absolute/path/to/network/deployment/model.onnx
Stack.transportPrimaryOnnxShadow=true
Stack.transportPrimaryOnnxSecondaries=false
Stack.transportPrimaryOnnxApplySigmoid=false
Stack.transportPrimaryOnnxOutputIndex=0
Stack.transportPrimaryInvert=false
Stack.pruneKine=false
Stack.storeSecondaries=true
```

Join these with semicolons in the simulation config-key argument. Alternatively,
use `Stack.transportPrimaryOnnxCCDBPath` instead of ModelFile, never both.
No threshold is required in shadow mode. It ignores the decision threshold and
keeps all tracks, including invalid model inputs/scores. The model runs at
PreTrack with the same birth features as pruning mode. Geometry navigation uses
a private navigator, as in the parent branch.

Default scoring covers independently simulated input tracks (including injected
secondaries), not generator-history entries that never reach PreTrack. Set
`transportPrimaryOnnxSecondaries=true` to also score transport-created secondaries.
The September 29 models were trained on input roots: secondary scores are an
exploratory evaluation, not a validated deployment scope.

Scores are persisted as `MCTrack.mTransportModelScore` in the Kine ROOT output;
`mTransportModelShadow` marks evaluated shadow tracks. A score of -1 means not
evaluated; NaN means evaluated but invalid/outside the selected feature domain.
Both fields survive MCTrack copies, reordering and parallel hit merging. MCTrack
schema version is incremented to 9; rebuild all consumers together.

The existing hit masks and first-mother links provide truth after full simulation.
`analyzeTransportShadow.py` saves score and recursive truth together in CSV.
Ground truth is deliberately computed after the event completes, not at PreTrack.
Full kinematics can be much larger than normal filtered output.

## Prepared models on Hydra

Base:
`/lustre/alice/users/csonnab/cern-fellowship/run/simnet/NN/output/29-09-2026/sim_prune_roots_qa_v3`

Each of `bdt`, `nn_deep64_tanh`, and `transformer_light` now has
`network/deployment/model.onnx`, `model.json`, and `model.qa.json`.
The original exports and weights are preserved. Adapters gather selected features,
apply the saved domain checks/scaling, and output one class-1 probability. They
were compared against original exports on 4,096 real rows each: maximum absolute
score error below 5e-7, zero saved-threshold decision differences on those rows.
This is numerical export QA, not a substitute for fresh simulation validation.
Reproducible conversion helper:
`/lustre/alice/users/csonnab/cern-fellowship/run/simnet/NN/src/training/export_shadow_models.py`.

## Prepare and submit the user's validation job

`submitLarge.py --onnx-model ...` selects `validation_data`, including the edited
500-event job. Without this option its previous training_data selection remains.
Separate model/mode names are added to the simulation output directory. Use a
separate generated-config folder for each model. For example, on Hydra:

```bash
cd /lustre/alice/users/csonnab/PhD/jobs/simulation
models=/lustre/alice/users/csonnab/cern-fellowship/run/simnet/NN/output/29-09-2026/sim_prune_roots_qa_v3
python3 submitLarge.py --submit 0 \
  --onnx-model "$models/nn_deep64_tanh/network/deployment/model.onnx" \
  --onnx-mode shadow \
  --write-configs-folder misc/sim_configs_onnx_shadow_nn_deep64_tanh
```

Inspect the generated config. After rebuilding and selecting the new environment,
repeat with `--submit 1` to create/submit the job. Change the model directory for
BDT or Transformer. Add `--onnx-secondaries` if you want scores for transport-created
secondaries too; this gets a distinct output directory. No jobs were submitted by
this implementation.

**Environment:** at implementation time `O2PDPSuite::latest-o2` explicitly loaded
`O2/feature-onnx-primary-pruning-local1`. Rebuild/update the suite to use the new
branch, or supply `--o2-env` with the appropriate rebuilt environment. The helper
preserves your existing container, partition, resources and reconstruction options.
It does not fix cluster resource availability or update an installed environment.

## Extract truth and make ROC curves

After the job completes, pass the actual signal Kine file(s) from ONE model run:

```bash
apptainer exec /lustre/alice/users/csonnab/TPC/TPC_PRODUCTION/Containers/rocm_torch_env.sif \
  python3 /scratch/alice/csonnab/MyO2/O2/Detectors/Base/macros/analyzeTransportShadow.py \
  --input /absolute/path/to/tf1/sgn_Kine.root \
  --out /absolute/path/to/shadow_qa --target-hit-loss 0
```

`--input` accepts multiple paths. Do not mix model runs or duplicate events.
The analyzer requires the new score fields and refuses inhibited/pruned inputs.
It propagates detector-hit masks over first-mother transport ancestry, supports
unordered indices and rejects cycles/bad indices. It stops propagation at each
input root: generator ancestry between independently simulated inputs is not a
transport-descendant relation. This distinction can differ from legacy labels
that propagated through generator-history links.

Outputs:

- `predictions.csv`: one row per scored track, score, validity, input-root flag,
  `can_avoid_subtree` (1 means no hits anywhere in its tree), and subtree hit mask.
- `roc.png` / `roc.pdf`: removal of hit-free trees against loss of hit-producing
  trees; invalid scores always stay kept and remain in the denominators.
- Per-split threshold scans and conventional ROC CSVs for valid scores only.
- `summary.json`: coverage, calibration/test event membership, chosen threshold,
  and independently measured test rates. AUC is restricted to valid scores.

Whole events are shuffled with seed 42 and split 50/50 into calibration/test.
The cut maximizes hit-free-tree rejection subject to the requested calibration
hit-tree-loss bound. It is never chosen on the held-out test events. More events
are required if either split lacks a class; a single event produces diagnostics
only. Zero observed loss is not a guarantee of zero future loss. These are tree
counts, not individual-hit loss or CPU savings. All scored secondaries are
included when enabled, so their trees overlap; do not sum these as unique saved
transport work.

## Enable actual rejection later

Use `--onnx-mode prune --onnx-threshold CUT` with the cut from `summary.json`,
keeping the same model and scoring scope. The runtime rejects valid scores >= CUT;
invalid scores stay kept. The submission helper refuses secondary pruning for a
root-only model contract. The shadow mode is explicitly disabled in prune mode.
Changing the threshold does not require rebuilding or retraining.

## Validation status

- C++ syntax checks passed for Stack.cxx and testStack.cxx using the Hydra build flags.
- Six Python tests passed, including synthetic ROOT -> truth -> ROC and event separation.
- C++ score-copy and ROOT-roundtrip tests were added to testStack.cxx; these need
  the rebuilt libraries to execute.
- No rebuilt binary simulation or cluster submission has been performed yet.
