Primary-pruning validation of existing O2 shadow CSVs

Installed on Hydra at:
/scratch/alice/csonnab/MyO2/O2/Detectors/Base/macros/

No O2 rebuild is needed for these Python scripts. Keep analyzePrimaryPruning.py
and analyzeTransportShadow.py together if copying them elsewhere.

FILTER ONE FILE (standard-library Python, one record in memory):

TOOLS=/scratch/alice/csonnab/MyO2/O2/Detectors/Base/macros
CSV=/absolute/path/to/tf1/sgn_transport_scores.csv
python3 "$TOOLS/filterTransportRoots.py" \
  --input "$CSV" --output "${CSV%.csv}.roots.csv.gz"

Use --max-rows 100000 and a separate sample output name for a smoke test.
The filter preserves every column value on is_simulation_root=1 rows, including
unscored input roots. It copies already-computed subtree truth unchanged.
It accepts CSV or gzip input; output compression is chosen by .gz suffix.
Existing outputs are never overwritten. Temporary output is published atomically
on success and removed on handled errors. Use completed, stable source files;
the filter rejects a file whose size or modification time changes during reading.
A killed process can leave a uniquely named .partial file; this is not a finished
output. No resume mode is implemented. Progress is printed every million rows.

ANALYZE ONE MODEL (original full CSV or filtered CSV/gzip):

apptainer exec -B /lustre -B /scratch \
  /lustre/alice/users/csonnab/TPC/TPC_PRODUCTION/Containers/rocm_torch_env.sif \
  python3 "$TOOLS/analyzePrimaryPruning.py" \
  --csv "${CSV%.csv}.roots.csv.gz" --out /absolute/path/to/primary_qa \
  --chunk-size 50000

To analyze the original directly, use --csv "$CSV". Multiple CSV paths may be
passed for DIFFERENT events/timeframes of ONE model. Do not pass both a full
file and its filtered copy: that would count the events twice. Run models separately.

Analysis reads only required columns in chunks. It retains scored ROOT scores,
truth and event-group IDs for exact ROC and threshold calibration: RAM scales
with the chunk plus the scored-root population, not the secondary population.
This is not a constant-memory approximation to ROC. Reduce --chunk-size if needed.

Analysis outputs roc.png, roc.pdf, threshold scan and valid-score ROC CSVs, and
summary.json. Whole events are split into calibration and held-out test sets.
The configured threshold is also evaluated; invalid scores remain KEEP decisions.
Use --max-rows 100000 for a bounded input smoke test (per CSV). This may stop in
the middle of an event and is NOT suitable for choosing a production threshold.
The summary marks such direct partial-input analyses as smoke tests. If analyzing
a separately filtered sample, also pass --max-rows so that the summary is marked.
A sample containing only one event produces descriptive QA, without selecting a cut.

PROCESS ALL THREE EXISTING RUNS (sequentially; full scans can take hours):

TOOLS=/scratch/alice/csonnab/MyO2/O2/Detectors/Base/macros
BASE=/lustre/alice/users/csonnab/PhD/jobs/simulation/sim_data/validation/o2sim_14102026_anchoredMC_24arp2_559843/SC/0_100
for RUN in "$BASE"/*_shadow_alltracks; do
  CSV="$RUN/tf1/sgn_transport_scores.csv"
  python3 "$TOOLS/filterTransportRoots.py" \
    --input "$CSV" --output "${CSV%.csv}.roots.csv.gz" || break
  apptainer exec -B /lustre -B /scratch \
    /lustre/alice/users/csonnab/TPC/TPC_PRODUCTION/Containers/rocm_torch_env.sif \
    python3 "$TOOLS/analyzePrimaryPruning.py" \
    --csv "${CSV%.csv}.roots.csv.gz" --out "$RUN/primary_qa" || break
done

The original CSV files are never deleted or modified.

PROGRESS AND EXACT ROW PERCENTAGES

Keep csvProgress.py alongside both scripts. Both tools now report scanned/total
rows and percentage, normally every million input rows. Analysis reports at chunk
boundaries; the default 50,000-row chunks align with million-row updates.
Before the first scan, exact CSV records are counted in a separate streaming pass.
Counting progress reports rows counted and approximate percentage of file bytes
read (compressed bytes for gzip). This extra pass can be expensive on large files.
The total is cached beside the source as <filename>.rows.json and reused while
file size, modification time and inode are unchanged. Bounded smoke tests count
only up to their requested limit; their percentage refers to that bounded scan.

STOP AFTER A MILLION-ROW BUDGET

Add --max-million-rows 10 to either filterTransportRoots.py or
analyzePrimaryPruning.py to stop after 10,000,000 input data rows per file.
Fractions are accepted (0.5 means 500,000 rows). Do not combine it with --max-rows.
Counting is limited to the same prefix, so the rest of the file is not scanned.
Progress percentage is relative to the bounded sample (or EOF if earlier).
The analysis saves its normal outputs and marks them as a partial-input sample.
The stop can fall inside an event; it is not an unbiased full-event sample.
Use a distinct output name for filtered samples, for example .first10M.roots.csv.gz.

CACHED SCORE AND PHYSICS QA

The updated analyzePrimaryPruning.py automatically saves selected_tracks.npz in
its output directory and generates extra_qa/. Keep primaryPruningCache.py and
plotPrimaryPruningQA.py with the existing scripts when copying them elsewhere.

The cache holds ONLY selected scored roots: scores, full-subtree truth, event IDs,
and available PDG/process/momentum/vertex/subtree-size/hit-mask fields. Invalid
scores are retained. This adds RAM proportional to those selected-root fields,
not to all secondary rows. Caches use compressed NumPy arrays without pickle.
They preserve the original row limit and partial-sample flag.

Additional QA includes score distributions by truth class, probability reliability,
valid-score Brier and binned calibration error, and count/validity/rejection tables
and plots by event, PDG, pT, momentum, pseudorapidity, vertex and subtree size.
All these plots describe the cached sample including calibration events; held-out
validation remains in the original summary.json. Plots cap PDG to the 20 most
frequent species and events to the first 50; the CSV tables include all categories.

Re-run all analysis and QA WITHOUT opening original CSVs:

apptainer exec -B /lustre -B /scratch "$CONTAINER" \
  python3 "$TOOLS/analyzePrimaryPruning.py" \
  --cache "$RUN/primary_qa_first10M/selected_tracks.npz" \
  --out "$RUN/primary_qa_cached"

To regenerate only the extra QA with a specific fixed threshold:

apptainer exec -B /lustre -B /scratch "$CONTAINER" \
  python3 "$TOOLS/plotPrimaryPruningQA.py" \
  --cache "$RUN/primary_qa_first10M/selected_tracks.npz" \
  --out "$RUN/primary_qa_extra" --threshold 0.99

Do not pass row limits with --cache: the cache already represents the selected
sample. The independent extra-QA script does not tune a threshold. Its --threshold
is optional; without it only coverage/count/score QA is produced.

Runs made before this update did not save the required data. To create a cache,
run the updated analysis once on the chosen CSV prefix or an existing filtered CSV.
Once created, new QA and changed threshold settings need only the cache.
