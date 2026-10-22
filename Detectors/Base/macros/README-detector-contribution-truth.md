# Detector-contribution truth and conservative pruning

Old `MCTrack::getHitMask()` is not sufficient for arbitrary-secondary pruning. FT0 optical hits carry parent labels without a photon hit bit, while PHOS and EMCAL/DCal aggregate energy under shower ancestors.

This branch records the actual executing contributor in an independent MCTrack mask. Normal `Stack::addHit` also marks contributions. FT0 marks every accepted hit including zero-Edep optical signal records; calorimeters mark every positive deposit before any hit aggregation. Legacy hit counters, track retention and detector hit contents are unchanged.

## Produce complete new truth

Build this branch and select that build in the runtime environment. Use:

```
Stack.recordDetectorContributions=true
Stack.pruneKine=false
Stack.storeSecondaries=true
Stack.transportPrimary=all
```

Alternatively use ONNX shadow mode to save scores without rejecting tracks. Old models are permitted in shadow mode only. CSV v2 requires contribution recording. Complete-truth mode rejects configuration combinations that would remove tracks. MCTracks default to contribution version zero; only the explicit unpruned recording mode stamps version one. The independent fields survive copying, ROOT persistence and event merging. ROOT MCTrack schema is 10.

## Hard pruning protections

The default protections are:

```
Stack.transportPrimaryOnnxProtectOpticalPhotons=true
Stack.transportPrimaryOnnxProtectSecondaries=true
```

Protection bit 1 means optical photon (50000050/50000051); bit 2 means transport-created secondary. A protected track is transported regardless of score. Requested birth scores are still evaluated and saved. Independent simulation input tracks, including externally injected particles, are roots even if they carry generator-history mother links.

Hard pruning additionally requires ONNX metadata `simnet.label_schema=detector-contribution.v1`. Do not disable secondary protection merely because training completed: first compare full detector response in independent shadow/controlled validation. Roots are still subject to statistical model errors; these rules are not a zero-loss guarantee. A geometry-only eta/phi/radius veto does not repair missing truth and is not a substitute for these protections.

## CSV and training schemas

CSV v2 replaces hit-mask truth columns with `own_contribution_mask`, `subtree_contribution_mask`, `own_has_contributions`, and `subtree_has_contributions`. It keeps `can_avoid_subtree` as the zero-contribution target. Additional columns preserve the legacy own-track mask and record contribution version, protection bits, raw model rejection and effective guarded rejection. `would_reject` remains the model-only decision; `effective_would_reject` applies protections. Missing thresholds give -1 for both decisions. The QA reader reports both model-only curves and effective guard counts. CSV v1 remains readable only as explicitly legacy diagnostic truth.

The new balanced training extractor writes label schema 10 and rejects missing/uninstrumented contribution branches. It includes roots and secondaries with shared features and omits nested contribution-free descendants. It resolves each source file's own birth geometry. Do not relabel old ROOT/CSV files by changing their version numbers: individual merged-signal contributors cannot generally be reconstructed from them.

## Prepared continuation

On the shared filesystem:

```
/lustre/alice/users/csonnab/PhD/jobs/simulation/misc/contribution_truth_v5_20261022
```

After the rebuild, run `submit_truth_jobs.py` on **virgo4**. Override `CONTRIBUTION_O2_ENV` if necessary to select the rebuilt O2PDPSuite module. It checks the installed MCTrack dictionary and Stack parameters before submitting anything, then submits two five-event baseline signal-transport jobs (seeds 2044/2045), with FT0/FV0/FDD and EMCAL/PHOS included. It does not run QED background simulation, digitization or reconstruction and does not export all-track CSVs. A dependent extraction/preflight job creates a bounded 600,000-row dataset and writes `TRAINING_READY.json` only after confirming matching features, rows, event-separated splits and test IDs for all three models. If fewer independent events survive, collect more events rather than bypassing the preflight.

Only after that ready marker exists, run `start_training.py` on **hydra**, the GPU-training cluster. It verifies the frozen dataset hash and submits all three `config_contribution_*.json` models through the existing create/run jobs scripts. The Transformer retains its added fully connected head. Historical model/run snapshots are preserved; no training has been launched against obsolete data.

## Validation

Changed ML/Stack/detector translation units were syntax-checked against the O2 build flags. Standalone tests cover previously invisible contributions, ancestry propagation, protection rules and rejection of uninstrumented truth. Seven CSV/ROOT-to-ROC tests pass. The extractor was compiled and exercised on a 400-track ROOT fixture with zero legacy hit bits, correctly separating 200 signal contributors from 200 avoidable tracks. Training rejects legacy/inconsistent labels and exported ONNX files carry the new truth contract.

A full rebuild must run the MCTrack ROOT round-trip test and DetectorContributionTruth test. Real detector-response validation and fresh production data still require that rebuilt runtime; syntax checks and synthetic fixtures do not replace them.
