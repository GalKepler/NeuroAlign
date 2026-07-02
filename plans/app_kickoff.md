# NeuroAlign Brain Explorer — Next Step Brief
Scope of this step: centroid extraction → literature enrichment → interactive viewer. Not the full CLAUDE.md; treat this as the prompt that kicks off this phase of the project.

## 1. Centroid + literature pipeline (build first, offline, one-time)
Run the centroid script (voxel-mean → MNI mm via affine) on both schaefer_400.nii.gz and the Tian volume. Output: region_centroids.json, keyed by atlas label ID, { "schaefer_017": [x, y, z], ... }.
[Guessing] Spot-check a handful of centroids against the atlas visually before running literature lookups on all 432 — cheap now, expensive to redo once it's baked into every region's panel copy.
Feed centroids to Neurosynth/NeuroQuery via Coord2Region (or direct API if Coord2Region's abstraction doesn't fit). Output: region_literature.json, keyed the same way, holding top associated terms + study links per region.
Merge centroids, literature, and your existing plain-language label lookup into one canonical region_reference.json. This becomes static, versioned, shipped with the app — no runtime calls to Neurosynth/NeuroQuery.

Contract to enforce: every region key in region_reference.json must match the region keys your regional_stacker BAG output uses. Fail loud at build time if any atlas label has no BAG counterpart or vice versa — don't let it fail silently at render time as a missing panel.

## 2. Session data

Per-participant JSON keyed by subject_code, schema as previously defined (BAG, cohort percentile, driver/deviant flags per region).
Login: subject_code only, for now. [Certain] This is a known, accepted gap — not solved in this step, flagged for a later auth pass before this goes anywhere beyond a controlled pilot.

## 3. Rendering (NiiVue)

Load Schaefer surface/volume + Tian volume together (whichever form you're actually shipping — you said Schaefer is volumetric MNI, so both atlases can load as NIfTI volumes with setColormapLabel).
Diverging indexed colormap, fixed range across all participants (confirmed decision — don't rescale per-participant).
Click/crosshair callback → resolve region ID → look up in both region_reference.json (static) and the loaded participant JSON (dynamic) → render panel.

## 4. Panel content, in order

Plain-language region name
This participant's BAG value + cohort percentile visual
Driver / deviant badges, if applicable
Neurosynth/NeuroQuery top terms for that region, with links to source studies

## 5. Build order for Claude Code

1. Centroid script + spot-check
2. Literature enrichment pipeline → region_reference.json
3. Static single-participant demo: mesh/volume load, fixed colormap, verify a known-high-BAG region actually renders hot
4. Click → panel wiring, still hardcoded participant
5. subject_code session loader
6. Polish pass