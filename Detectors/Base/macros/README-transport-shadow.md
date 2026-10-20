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
Stack.transportPrimaryOnnxCsv=auto
Stack.transportPrimaryOnnxCsvRootsOnly=true
Stack.transportPrimaryOnnxModelLabel=nn_deep64_tanh
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

The prepared track_pruning settings score only simulation input roots. They set
transportPrimaryOnnxCsvRootsOnly=true, exporting all input-prefix rows (including
unscored generator-history entries) and omitting transport-secondary rows.
Full descendant hit masks and counts are computed before filtering, so root truth
is identical to selecting roots from an all-track CSV. Root means simulation input,
not the physics definition of primary. Generator-history entries that never reach
PreTrack are not evaluated; metrics select model_evaluated == 1.

For all-track diagnostics, set transportPrimaryOnnxCsvRootsOnly=false (the default).
Secondary scoring is independently controlled by transportPrimaryOnnxSecondaries;
turning secondary scoring off alone does not reduce CSV rows. Retain pruneKine=false
and storeSecondaries=true for complete truth: roots-only CSV reduces text output,
not the size of the full Kine ROOT file. Rebuild O2 before using the new parameter.

Scores are persisted as `MCTrack.mTransportModelScore` in the Kine ROOT output;
`mTransportModelShadow` marks evaluated shadow tracks. A score of -1 means not
evaluated; NaN means evaluated but invalid/outside the selected feature domain.
Both fields survive MCTrack copies, reordering and parallel hit merging. MCTrack
schema version is incremented to 9; rebuild all consumers together.

The hit merger computes truth after a complete event is merged and writes one CSV
for that simulation output prefix: for example `tf1/sgn_transport_scores.csv`.
It uses the existing hit masks and first-mother transport links. Workers do not
write separate CSVs. Separate model jobs/timeframes retain their own CSV files;
do not point concurrent jobs at one shared filename. This automatic writer is for
the parallel o2-sim hit-merger workflow used by submitLarge.py; serial simulations
can still use the ROOT-to-CSV analysis path below.
Zero-hit events are retained when CSV output is active, even if the usual event
filter was requested, so they are represented in the evaluation. Every complete
event is flushed to disk; an interrupted run contains only the portion written.
The schema is fixed and rows are in merged event/track order, with no downsampling.
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

`submitLarge.py` contains three explicit dictionaries in `track_pruning`, one each
for BDT, Deep NN and Transformer. `dict_use = track_pruning` selects all three.
Each dictionary contains its full deployment model path and shadow options in
SIM_OPTIONS, secondary scoring disabled, roots-only automatic CSV output, a short model label,
and its saved root-validation threshold as an initial reference cut. Shadow mode
ignores that cut for transport; it only supplies the CSV's would_reject column.
Output folders end in `<model>_shadow_roots`. The original command
line and submission loop are retained; there are no added model command-line flags.
All three retain the requested cap of 500 events and seed 1043. The anchored
collision context may override the event count (114 in the inspected run).

On Hydra, generate configs for inspection without submitting:

```bash
cd /lustre/alice/users/csonnab/PhD/jobs/simulation
python3 submitLarge.py --submit 0 --write-configs-folder misc/sim_configs_track_pruning
```

After rebuilding and updating the environment in config_sim.json, repeat with
`--submit 1` to submit the three jobs. Change the literal dictionaries to select
other models/options or reduce the list to the jobs you want. Primary-only scoring and CSV export are enabled in all three settings.

**Environment:** `O2PDPSuite::latest-o2` currently loads
`O2/feature-onnx-primary-pruning-keep-with-threshold-local1`. Rebuild O2 in this
environment before using transportPrimaryOnnxCsvRootsOnly; the installed binary
must know the new parameter. No jobs were submitted for this change.

## Read the single simulation CSV

No ROOT extraction is needed for the new jobs. Read `sgn_transport_scores.csv`
in the relevant timeframe directory directly with pandas or another CSV tool.
The file includes input roots with the prepared settings (every stored track when
CsvRootsOnly=false); select **model_evaluated == 1** for model
metrics. Unscored generator entries preserve complete IDs/context in the same file.
Invalid scores remain in this selected population as KEEP decisions.

Key columns:

| Columns | Meaning |
|---|---|
| model, run_prefix, event_id, event_entry, track_id | Model and final merged track identity; event_entry is the zero-based Kine entry, event_id is the simulation event ID |
| mother_id, transport_mother_id | Original first mother and transport parent; input roots have transport_mother_id=-1 even if they have a generator mother |
| simulation_root_id, depth | Independently simulated ancestor and number of transport-parent edges; -1 root ID means no input ancestor |
| is_simulation_root, is_transport_secondary | Input-prefix membership versus a transport-created secondary; this is not the physics definition of primary |
| hitless_tree_root_id | Highest ancestor whose entire transport subtree is hit-free, including the track itself; -1 if this track's subtree has a hit |
| is_hitless_tree_root, is_hitless_tree_descendant | First track of a wholly hit-free tree versus a nested daughter; a secondary may be a hitless-tree root |
| n_descendants | Number of stored transport descendants, excluding this track |
| model_evaluated, valid_score, score | Model execution at PreTrack, score validity, and class-1 score; -1 means unscored, NaN means invalid |
| can_avoid_subtree | Ground truth: 1 only if this track AND every transport descendant have no recorded detector hit |
| own_hit_mask, subtree_hit_mask, own_has_hits, subtree_has_hits | Detector-hit masks and boolean hit truth, not counts of individual hits |
| parent_can_avoid_subtree | Parent subtree truth, or -1 for no transport parent |
| deduplicated_candidate | Scored hit-producing trees plus only the first track of each hit-free tree |
| configured_threshold, would_reject | Initial reference cut and hypothetical decision; -1 if no applicable threshold/evaluation |
| pdg, process, px/py/pz, vx/vy/vz, t_ns, weight | Birth properties for phase-space checks; coordinates in cm, momenta in GeV/c, time in ns |
| shadow_mode, csv_schema_version | Full-transport provenance and CSV layout version |

The independent-root boundaries deliberately exclude generator-history edges from
transport truth. Missing/out-of-range mother indices and cycles abort CSV generation
rather than inventing labels. A fully hit-free chain A -> B -> C has hitless root A
for all three; only A has is_hitless_tree_root=1.

## Make ROC curves and choose a threshold directly from CSV

```bash
apptainer exec /lustre/alice/users/csonnab/TPC/TPC_PRODUCTION/Containers/rocm_torch_env.sif \
  python3 /scratch/alice/csonnab/MyO2/O2/Detectors/Base/macros/analyzeTransportShadow.py \
  --csv /absolute/path/to/tf1/sgn_transport_scores.csv \
  --out /absolute/path/to/shadow_qa --scope roots --target-hit-loss 0
```

The CSV is the sole data input. Use `--scope deduplicated` to omit nested hit-free
daughters, or `--scope roots` to recover input-root-only QA. Multiple CSV paths
are supported for one model; different models and duplicate file paths are rejected.
For older shadow runs without automatic CSV, use `--input .../sgn_Kine.root` instead.
That fallback writes a reduced predictions.csv from ROOT and supports the same scopes.

Outputs are `roc.png`, `roc.pdf`, threshold-scan/ROC CSVs and `summary.json`.
The summary includes the configured cut's observed rates, coverage and invalid
scores, plus a newly selected threshold and independent test performance.
Whole events are shuffled with seed 42 and split 50/50 into calibration/test.
The cut maximizes hit-free-tree rejection subject to the requested calibration
hit-tree-loss bound. It is never chosen on held-out test events. More events are
required if either split lacks a class. Invalid scores always stay kept and remain
in the operational denominators; conventional ROC/AUC uses valid scores only.
Zero observed loss is not a guarantee. Tree counts are not individual-hit loss or
CPU savings. All-track subtrees overlap: do not sum descendant counts as unique
saved transport work, or treat every nested hypothetical rejection as independent.

## Enable actual rejection later

In the desired setting's SIM_OPTIONS, set Stack.transportPrimaryOnnxShadow=false
and add Stack.transportPrimaryOnnxThreshold=CUT using the calibration cut from
summary.json. Keep the same model and scoring scope, and change the output folder
for the pruning run. Valid scores >= CUT reject; invalid scores stay kept.
Root-trained models require validation before enabling secondary pruning.
Changing the threshold does not require rebuilding or retraining.

## Validation status

- C++ syntax checks cover Stack.cxx, testStack.cxx and O2HitMerger.cxx using the Hydra build flags.
- Seven Python tests passed, including exact roots-only/all-track CSV root-row
  equivalence, descendant-hit preservation, and compiled C++ CSV writer -> CSV-only ROC,
  all/root/deduplicated scopes, synthetic ROOT input and event separation.
- The standalone C++ test also checks unordered ancestry, 100,000-track deep trees,
  generator-root boundaries, nested hitless roots and malformed ancestry.
- C++ score-copy and ROOT-roundtrip tests were added to testStack.cxx; these need
  the rebuilt libraries to execute.
- No rebuilt binary simulation or cluster submission has been performed yet.

## Primary mother context fix (15 October 2026)

Parallel transport preserves event-global generator mother IDs when it splits
primaries into chunks. Those IDs cannot be looked up in a worker-local Stack.
The old PreTrack inference path did exactly that: an ID outside the local primary
array became missing, while an ID inside it could select an unrelated particle.
The selected `mother_abs_pdg` / `has_gen_mother` inputs then became NaN or incorrect.

The BDT first-event root sample demonstrates both failures with 500-particle
chunks: 33,924 scored roots have NaN scores and mothers outside the local primary
array; 10,821 scores are finite, but reconstructing the local lookup gives the
wrong mother PDG for 9,344 of those. All mothers exist in the full input event.
These old scores and their QA curves must not be used to validate model quality.

PrimaryChunk version 2 now carries one full-event mother PDG per input particle.
The primary server resolves it before splitting; O2MCApplication passes it to the
Stack, which uses it only for simulation roots. Serial/full-event initialization
retains a full-primary-array lookup; secondary transport keeps its local ancestry
lookup. Original mother IDs and transport truth remain unchanged. Workers reject
missing/mismatched chunk context. Remaining invalid ONNX outputs log bounded
examples with raw nonfinite feature names (these can include unused inputs).

Rebuild the simulation data dictionaries, DetectorsBase, Steer, primary server,
and workers together before running a fresh small shadow validation. Do not mix
old and new binaries. Existing CSV/NPZ scores do not change when source is fixed;
regenerating plots from those caches cannot repair inference inputs. Full-transport
truth remains useful, but rescoring would require reconstruction of birth features.

Validation of the mother-context fix: the complete rebuilt MCStack test suite
passed, including new chunk-context/reset and ROOT serialization tests. Syntax
checks passed for the primary server, worker, application and tests. A controlled
C++ feature replay through all three deployed ONNX graphs reproduces NaN for
missing mother context and finite scores after correction. The production
installation has not been replaced, and no new simulation jobs were submitted.
