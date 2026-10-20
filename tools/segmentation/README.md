# Roughness Segmentation

Generate SAM3-derived semantic color-ID maps for roughness evaluation and
segmentation-conditioned diagnostics. The SAM3 SDK, checkpoints, input images,
and the full generated mask collection are obtained separately. The
[runnable example](../../examples/README.md) already includes its frozen region map.

## Layout

```text
segmentation/
├── generate_roughness_segmentation.py  # Manifest-based command-line entry
├── scene_inference.py                 # Full-image and crop-based proposals
├── scene_inference_contextual.py      # Scene-profile-aware proposals and repair
├── region_merging.py                  # Region competition and boundary handling
├── configs/
│   ├── indoor.json                    # Standard vocabulary and thresholds
│   └── indoor_contextual.json         # Scene profiles, vocabulary, thresholds
└── SAM3_LICENSE                       # SAM3 dependency license reference
```

## Methods

Choose the method explicitly with `--method`:

- `standard`: `scene_inference.py` proposes and merges masks using the standard
  indoor vocabulary, layer priority, and gap filling.
- `contextual`: `scene_inference_contextual.py` proposes masks using scene-profile
  controls; `region_merging.py` resolves support-surface competition, image-border
  regions, holes, and unassigned areas using the contextual repair helpers.

Record the method and configuration used to generate each region map. Use the
same frozen maps across methods when comparing region-based metrics.

## Dependencies

Install the Meta SAM3 SDK and obtain an image-model checkpoint separately under
its original license and access terms. The custom workflows require `numpy`,
`Pillow`, `torch`, and the SDK. Install `scipy` for contextual region repair.
`SAM3_LICENSE` is a dependency license reference, not a replacement license for
the entire project.

## Usage

First check input paths without loading SAM3 or generating any maps:

```bash
python3 tools/segmentation/generate_roughness_segmentation.py \
  --manifest data/manifests/mainaxis/roughness_manifest_mainaxis.csv \
  --data-root /path/to/openroomff_mainaxis \
  --output-root /path/to/segmentation/openroomff_mainaxis \
  --method standard --dry-run
```

Generate using an explicit local checkpoint:

```bash
python3 tools/segmentation/generate_roughness_segmentation.py \
  --manifest data/manifests/ablation/roughness_manifest_ablation.csv \
  --data-root /path/to/ablation_R \
  --output-root /path/to/segmentation/ablation_R \
  --method contextual --checkpoint /path/to/sam3.pt --cuda-device 0
```

`--sam3-repo` optionally selects an external SDK checkout. `--cuda-device` indexes
visible GPUs. The inference modules read `SAM3_*` environment controls before
inference; those settings are captured in the generated run record. Use
`--max-images 1` for a small run. Existing maps are skipped unless `--overwrite`
is specified. Keeping a separate output root avoids modifying dataset assets.

For ablation, the roots must contain `openroomff_mainaxis/` and
`openroomff_stresstest/` prefixes. The runner maps raw manifest names
`N_im_5.rgbe` to existing processed `N_im.png`; it does not perform tone mapping.

## Outputs and evaluation

The pipeline saves proposal masks, `summary.json`, `scene_label_map.png` (class
IDs), `scene_label_map_color.png` (RGB class colors), and `label_ids.json` under
`_sam3_details/`. The color-ID map is copied to the manifest's
`segmentation_path`, typically `N_seg.png`. A per-image `generation_record.json`
records the chosen method, checkpoint path, manifest/output hashes, and settings.

Use the pure color-ID map for evaluation, not an RGB overlay. Colors represent
semantic classes: disconnected pixels with the same color form one evaluator
region. The valid mask is separate; black/unassigned pixels inside it contribute
a region in the current evaluator. Preserve exact colors with PNG and
nearest-neighbor resizing.

Pass the corresponding processed segmentation tree as `--region_dir`, select
`--region_source dir`, and use `--region_suffix _seg` when pair keys are `N`.
The evaluator does not automatically read the CSV segmentation column, and its
PBR ObjectSegmentation fallback does not discover OpenRooms `N_seg.png` files.
