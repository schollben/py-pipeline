# Files needing upstream investigation — PROCESSED/V1/

Generated from a read-only scan of all 36 `.h5` files. Every finding below is
visible from a plain `h5read`; no pipeline run is needed to reproduce them.

**Current state: 17 of 36 files load.**

Diagnostic fields used throughout: `stim_id`, `stim_properties`, `target_number`,
`photostim_triggers_sec`, `markpoints_laser_power`, `markpoints_group_info`,
`markpoint_assigned_roi`, `cyc`, `cyc_photostim_only`, `dff`,
`params/is_spontaneous`, `params/stim_file_num`.

---

## PRIORITY 1 — the 2026 sessions: stimulus extraction never ran

**10 files. Recordings look clean; the trial table is simply absent.**

The failure is bounded by acquisition date, not by anything about the individual
recordings:

| Session date | trial table built | not built |
|---|---|---|
| 11032024 | 11 | 0 |
| 12032024 | 4 | 0 |
| 07132025 | 1 | 0 |
| **06262026** | **0** | **5** |
| **07212026** | **0** | **5** |

Every 2024 and 2025 session built its table. No 2026 session did. All pipeline
params are identical across both eras — `is_2p_opto=1`, `photostim_ch=3`,
`visual_trigger_ch=2`, `vrec_sample_rate=10000`, `use_inference=1` — so this is
not a settings difference in the processing pipeline.

In every one of these files `params/is_spontaneous = 0` and `stim_file_num` is a
real index (1–6), meaning a visual stimulus file *was* specified at acquisition.
But `stim_properties`, `stim_id`, `target_number`, `target_trial`, and `cyc` are
all empty. The photostim side partly survived: `photostim_triggers_sec` is
populated with machine-regular intervals, and `cyc_photostim_only` was built.

**Interpretation:** the recordings are fine. The step that reads the stimulus log
and builds the trial table either could not find the log, or parsed it to nothing,
and failed silently rather than erroring.

### Files

| File | duration | ROIs | `stim_file_num` | psTrig | ISI | `cyc_photostim_only` |
|---|---|---|---|---|---|---|
| `06262026-1127-001` | 5.0 min | 32 | 1 | 9 | 29.98 s | 288 |
| `06262026-1127-003` | 5.8 min | 27 | 3 | 11 | 29.98 s | 297 |
| `06262026-1127-004` | 5.8 min | 27 | 4 | 10 | 29.98 s | 270 |
| `06262026-1127-005` | 5.7 min | 22 | 5 | 10 | 29.98 s | 220 |
| `07212026-1350-001` | 4.4 min | 64 | 1 | 119 | 2.00 s | 7616 |
| `07212026-1350-003` | 12.3 min | 77 | 3 | 239 | 3.00 s | 18403 |
| `07212026-1350-005` | 12.7 min | 62 | 4 | 239 | 3.00 s | 14818 |
| `07212026-1350-006` | 6.3 min | 64 | 5 | 119 | 3.00 s | 7616 |
| `07212026-1350-007` | 12.7 min | 60 | 6 | 239 | 3.00 s | 14340 |

(`06262026-1127-002` is listed separately below — same era, but nothing was
acquired.)

### What to check

1. **Do the raw stimulus logs still exist** on the stim computer for 06/26/2026
   and 07/21/2026? Do their filenames match what `stim_file_num` (1–6) implies?
   If they exist, these files are recoverable by re-running extraction alone.
2. **Did the stim-log path, naming convention, or output format change** between
   12/2024 and 06/2026? A new column layout that the parser doesn't recognise
   would produce exactly this — empty output, no error.
3. **Was the pipeline itself updated** between those dates in a way that moved or
   renamed the trial-table step's inputs?
4. **Why did extraction fail silently?** Whatever the root cause, the step should
   raise rather than write empty arrays — that is what made this hard to spot.

---

## PRIORITY 2 — 2026 sham groups have zero targets

**Independent of the above, and it will break analysis even after the trial
tables are rebuilt.**

Comparing `markpoints_group_info` (columns: `condition_idx`, `unique_group_id`,
`n_targets`, `dispersion`) across eras:

```
11032024-1313-012 [2024]  uid=[0, 0, 1]     n_targets=[10, 10, 10]   powers=[0, 91, 91]
07132025-1042-003 [2025]  uid=[0, 0, 1, 1]  n_targets=[4, 4, 4, 4]   powers=[0, 80, 0, 80]
07212026-1350-003 [2026]  uid=[0, 0, 1, 2]  n_targets=[9, 9, 0, 0]   powers=[0, 110, 0, 110]
07212026-1350-001 [2026]  uid=[0, 0, 1, 2]  n_targets=[3, 3, 0, 0]   powers=[0, 85, 0, 85]
```

Two problems in the 2026 files:

- **`n_targets = 0` for both sham conditions** (conditions 2 and 3). The 2024/2025
  shams carry the same target count as their real pair. A sham with no targets has
  no cells to address at 0 mW.
- **`unique_group_id` is `[0, 0, 1, 2]`** instead of the `[0, 0, 1, 1]` pattern
  that pairs each real condition with its sham. Conditions 2 and 3 each get their
  own uid, so neither pairs with anything.

The dispersion column for those sham rows is also `0.0`, where 2024/2025 shams
carry a large dispersion value (the defining sham signature).

**What to check:** whether the markpoints/group definition changed in the 2026
acquisition protocol, or whether this is a second symptom of the same extraction
failure. Note the loader now falls back to the session sham pool when a real group
has no sham sibling, so these will not be silently dropped — but a sham with zero
targets is still wrong and should be fixed at the source.

---

## PRIORITY 3 — spontaneous sessions (no action needed, listed for completeness)

**9 files.** `params/is_spontaneous = 1` and `stim_file_num = -1`: no stimulus
file was specified, so there is correctly no visual trial table. These are
legitimate spontaneous (± photostim) runs, not failures.

`07212026-1350-004`, `07212026-1350-008`, `11032024-1313-005`,
`11032024-1313-006`, `11032024-1313-016`, `12032024-0951-001`,
`12032024-0951-002`, `12032024-0951-004`, `12032024-0951-008`

Of these, `11032024-1313-006` is a substantial recording (18.7 min, 99 photostim
triggers at 5.00 s ISI) and may be worth analysing via the photostim-only path
once that is in place. The rest are short (0.1–5 min) and several have no
photostim triggers at all — likely abandoned at the start:

| File | duration | psTrig |
|---|---|---|
| `07212026-1350-004` | 0.6 min (1000 fr) | 0 |
| `07212026-1350-008` | 0.6 min (1000 fr) | 0 |
| `11032024-1313-005` | 0.1 min (108 fr) | 0 |
| `11032024-1313-016` | 0.1 min (240 fr) | 3 |
| `12032024-0951-002` | 5098 fr | 0 |
| `12032024-0951-008` | 2811 fr | 0 |

Confirm against acquisition notes, then archive.

---

## PRIORITY 4 — nothing acquired

| File | note |
|---|---|
| `06262026-1127-002` | `is_spontaneous=0`, 11 photostim triggers, but no markpoints metadata at all (`laser_power`, `group_info`, `xy_pix`, `markpoint_assigned_roi` all empty) and `cyc_photostim_only` empty. Same 2026 era as Priority 1 but more thoroughly empty. |
| `12032024-0951-001` | 855 frames, no triggers, no markpoints. |
| `12032024-0951-004` | 1369 frames, no triggers, no markpoints. |

---

## NOT A PROBLEM — `07132025-1042-002`

Visual-stimulus-only session. `is_2p_opto = 0` and no markpoints data, which is
correct: photostim was not run. Its visual table is complete and well-formed —
320 trials, 16 directions × 5 contrasts, exactly 4 trials per condition, 100 ROIs.
Loads and analyses correctly.

Its sibling `07132025-1042-003` from the same session has `is_2p_opto = 1` and
built everything, and is the reference file for full-protocol analysis.

---

## Files that load but carry known damage

### Trailing truncation — aborted runs

The stimulus table is written in full when a run starts; an aborted recording
leaves the trials that never played as surplus **trailing** rows in
`stim_properties`. Verified per file by the invariant that every trial sharing a
`stim_id` must carry the same `(direction, contrast)`: trimming the tail satisfies
it, trimming the head does not.

| File | `stim_properties` | `stim_id` | surplus |
|---|---|---|---|
| `11032024-1313-003` | 636 | 492 | 144 trials never ran |
| `11032024-1313-007` | 384 | 383 | 1 |
| `11032024-1313-014` | 384 | 383 | 1 |
| `11032024-1313-017` | 384 | 383 | 1 |

The loader now truncates to `len(stim_id)` and warns. No upstream action strictly
required, but the writer should ideally record the presented trial count rather
than the planned one.

### `11032024-1313-003` — PHOTOSTIM LABELLING UNUSABLE

Beyond the 144-trial truncation, `target_number` (635) **exceeds** `stim_id` (492)
by 143: photostim assignments were planned for trials whose presentation never
happened. `cyc_trial_group` warns and truncates to the shorter, but that
truncation does not restore the correspondence.

Checking which ROIs each group actually drives — cond 1 (`target_number`=2)
targets ROIs 0–4, cond 2 (`target_number`=3) targets ROIs 5–9:

```
tn     power     ROIs 0-4   ROIs 5-9
 1    0 (sham)    0.1110     0.1236     <- sham HIGHEST in both sets
 2    75          0.1008     0.1134
 3    75          0.0933     0.1068
```

The sham exceeds both real groups everywhere, and neither real group is selective
for its own targets. Shifting the alignment by 1 or 2 trials barely moves these
numbers — the labels are **decorrelated from the data**, not displaced. With a
143-entry gap out of 635, trial *k* stops corresponding to trial *k* at the first
divergence, so no uniform offset can repair it.

**Do not use this session for any photostim analysis** until the writer is fixed.
Its visual side is fine after the `stim_properties` truncation, so visual tuning
remains valid.

### Dropped first photostim TTL — `11032024-1313-001` and `-003`

In both files a real photostim event fired exactly one ITI before the first
recorded TTL, with no TTL saved for it. Detected from raw fluorescence: the PMT
shutter closes during photostim, so mean raw F dips sharply.

| File | inferred event | dip there | dip at known TTLs | nearby controls |
|---|---|---|---|---|
| `11032024-1313-001` | frame 769 (= 919 − 150) | 23.4% | median 23.8% | 4–6% |
| `11032024-1313-003` | frame 348 (= 408 − 60) | 35.5% | median 24.7%, 5–95% [11.6, 32.5] | −1 to 1.5% |

Both sit inside the known-TTL distribution; controls a few frames away are near
zero. Two ITIs back in `-001` gives 12.5%, below the population — so exactly one
trial was dropped, not more.

Corroborated in `-003` by the block structure: the first `target_number` block
runs 31 entries where every other runs 32, and the first `target_trial` block runs
95 where every other runs 96.

In `-001` the arrays are shifted rather than short: `photostim_triggers_sec` has
36 entries vs 35 for `stim_on_2p_frame`, and the last two land at frames 6001 and
6151 — past the 5975-frame recording — both clamped to 5974. The sequence
overruns the end by exactly what it is missing at the start.

**`-001` is recoverable** (one dropped leading TTL, otherwise self-consistent).
**Upstream:** stop dropping the first TTL when writing the `.h5`.

### `target_number` one short of `stim_id`

`07132025-1042-003` (240 vs 239) and `11032024-1313-012` (384 vs 383). This is the
spurious-first-TTL case that `dropFirstEvents` handles at runtime. Consistent
enough across files to look like convention — worth confirming it is intentional,
since `photostim_group_map` consumes `target_number`.

### Mismatch scan across all loading files

Only `-003` has a large gap; every other file is within ±1. Blocking is not itself
a problem — `-012` and `-007` are blocked and fine.

```
07132025-1042-003  240/239   -1  interleaved   ok
11032024-1313-012  384/383   -1  BLOCKED       ok
11032024-1313-007  383/383    0  BLOCKED       ok
11032024-1313-003  492/635 +143  BLOCKED       *** UNUSABLE ***
```

---

## Quick reproduction

```python
import h5py, glob, os
for p in sorted(glob.glob('PROCESSED/V1/*.h5')):
    with h5py.File(p, 'r') as f:
        pa = f['params']
        print(os.path.basename(p),
              'table:', 'YES' if f['target_number'].size else 'no',
              'spont:', int(pa['is_spontaneous'][()]),
              'sfn:', int(pa['stim_file_num'][()]),
              'psTrig:', f['photostim_triggers_sec'].size)
```

A file is a genuine anomaly when `table: no` and `spont: 0` — a stimulus file was
specified but no trial table was built. That is exactly the 10 Priority-1 files
plus `06262026-1127-002`.
