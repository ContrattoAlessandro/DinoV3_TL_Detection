# Data and annotation protocol

## DTLD

Use DTLD v2.0 native annotations and full-resolution RGB images. Obtain the data
under its terms through the [native parser repository](https://github.com/julimueller/dtld_parsing).
The expected local layout is:

```text
datasets/
  DTLD/v2.0/DTLD_train.json
  DTLD/v2.0/DTLD_test.json
  DTLD/... native city/session Bayer TIFFs ...
  DTLD_jpg/train/*.jpg
  DTLD_jpg/test/*.jpg
```

`scripts/preprocessing/convert_dtld.py` decodes Bayer TIFFs to full-resolution RGB
JPEGs through the attributed DTLD parser. No side-cropped image dump is required.
Both `crop_sides` and `label_crop_sides` are zero in the default configuration.

### Full-frame coordinates and masks

Images and boxes share one deterministic letterbox transform. With source size
$(H,W)$ and target $(H_t,W_t)=(720,1280)$, scale by
$s=\min(W_t/W,H_t/H)$, round the resized dimensions to $(H_r,W_r)$,
then center at integer offset $(o_y,o_x)$:

$$
x'=x\,W_r/W+o_x,\qquad y'=y\,H_r/H+o_y.
$$

During training, an optional zoom factor in [0.9,1.0] reduces the scale and
randomizes placement while retaining the entire original frame. At inference,
the transform is centered with zoom 1.0. Bicubic interpolation and RGB padding
(124,116,104) are shared by dataset and folder inference.

The 16-pixel patches form a 45×80 grid. A patch is content-valid if it intersects
the resized image; padding-only patches are excluded. Geometry uses content-relative
centers $(x,y)$, clipped to [0,1], and is encoded as $(x,y,2x-1,1-y)$.
Tiny lamp boxes receive at least one valid token when intersecting the content.

Lampness is the union of box supports. State, relevance, direction, and pictogram
have independent conflict masks; disagreement in one attribute does not discard
the others. A one-patch ignore band limits uncertain boundary background labels.
State, direction, pictogram, and lamp-region relevance losses average each lamp
instance and then each image, so large lamps do not dominate these attributes.
Lamp detection and background relevance average valid tokens within each image.
Agreeing overlapping boxes retain contributions
for each instance. Training all-lamp erasure clears corresponding targets and
updates the image label to NoR.

Native housing directions are back/front/left/right. State labels are
green/off/red/red-yellow/unknown/yellow. The label priority is relevant
red/yellow/red-yellow → RR, else relevant green → RG, else NoR. Images with only
relevant off/unknown lamps remain in NoR at full sample weight. The dataset
records this group as `pseudo_nor` for separate diagnostic reporting.

The official training split contains 28,525 usable images under this policy:
9,570 RR, 17,583 RG, and 1,372 NoR. Cross-validation partitions whole cities;
ordinary training creates a session-disjoint validation split within the
official training data. Neither loader uses official test data for training
or per-epoch checkpoint selection.

The official test split contains **12,453 images**: 4,359 RR, 7,569 RG, and
525 NoR, including 213 NoR frames with relevant off/unknown lamps only. All
native test entries are evaluated; no missing images or invalid-frame exclusions
occur. The audited RGB train/test dumps have zero shared image identities, shared
sessions, or exact JPEG duplicates. The audit covers 1,478 training sessions and
632 test sessions, with no exact duplicates within either split. Annotation and
ordered membership SHA-256 hashes are recorded in the
[frozen diagnostic plan](experiments/diagnostics/plan.json). The audit hashes the
prepared JPEG bytes, rather than asserting equivalence of every possible
conversion of the native TIFFs.
Session groups are the native `city/route/timestamp` folders parsed by `_seq_of`.

## ATLAS relevance annotations

The retained transfer benchmark contains 528 manually labeled images: an
initial collection of 28 and an additional collection of 500. The latter
contains 380 front-medium, 78 front-tele, and 42 front-wide images. These are
sampled driving frames, so correlated frames are not independent observations.

The frozen [annotation snapshot](../metadata/atlas_labels.json), revision 561,
contains RR/RG/NoR labels, uncertainty flags, image SHA-256 hashes, and relative
filenames. The [collection manifest](../metadata/atlas_collection.json) records
the exact membership and original/additional partitions. There are no exact
image duplicates in the 528-image collection. Image pixels are not redistributed
in this repository. Place the corresponding source images beneath
`datasets/atlas_relevance_expanded/`, preserving the manifest paths.

Labels use the image-level rule in the README. Annotation records retain
uncertainty rather than silently forcing ambiguous images into the certain-only
benchmark. Primary retained results include all 528 images (245 RR, 95 RG, 188
NoR); run evaluation with `--include-uncertain` to reproduce them. The evaluator
can also report the certain-only subset. It rejects missing files or changed
image contents before computing scores.

The local annotation tool initially hides supplied model predictions. This
supports blind labeling for new collections; the saved snapshot and annotation
history define the provenance of the retained benchmark. These manually assigned
labels should not be described as official ATLAS relevance ground truth.

## VZC-TLD

The source is
[vzc-research-chapter/vzc-traffic-light-dataset](https://huggingface.co/datasets/vzc-research-chapter/vzc-traffic-light-dataset),
pinned at revision `083bc5d626a2fe660171ba8a84c848222a661f02`.
The verified snapshot has 3,003 files, including 2,997 PNGs. Published COCO
annotations reference 2,988 images and 25,497 lamps. Nine unreferenced PNGs are
excluded from evaluation.

```sh
python scripts/download_vzc.py
python scripts/download_vzc.py --verify-only
```

The downloader uses the user's existing Hugging Face access. Complete any
required dataset access process on its source page. Verification checks file
size, available upstream LFS hashes, image integrity, and per-file SHA-256,
then writes `datasets/VZC_TLD/download_manifest.json`.

Eight COCO categories encode relevance and state:

```text
not_relevant_unknown, not_relevant_green, not_relevant_yellow, not_relevant_red,
relevant_unknown, relevant_green, relevant_yellow, relevant_red
```

The evaluator maps these to the same image-level class rule as DTLD. Keep all
annotations: **25,471 of 25,497 lamps have `iscrowd=1`**, so generic COCO crowd
filtering would discard almost all labels. Eight test and 26 train images have
no lamp annotations and map to NoR. Annotation and image hashes are verified
before inference; invalid references and conflicting duplicate labels are rejected.

| Published split | Images | RR | RG | NoR | NoR with relevant unknown |
|:--|--:|--:|--:|--:|--:|
| Test | 598 | 331 | 176 | 91 | 47 |
| Train | 2,390 | 1,350 | 735 | 305 | 130 |

The VZC *test* split is the primary transfer benchmark. The VZC *train* split is
also evaluated as a separate diagnostic; its name refers to the source dataset,
and no model training uses those images here. A strict sensitivity analysis
excludes relevant-unknown-only images, leaving 551 test images and 44 NoR images.

There are two exact duplicate image pairs: one crosses the published train/test
boundary and one lies within train. Their labels agree. No duplicates occur
within the test split. The evaluator reports unique-image sensitivity scores
and the overlap of 38 filename location prefixes across source splits. These
audits do not establish a fully independent sequence-level partition.

## Attribution and data terms

The helpers under `scripts/preprocessing/dtld_parsing/` derive from the native
DTLD parser and retain their upstream author attribution. Repository code uses the root
[AGPL-3.0 license](../LICENSE); dataset access and redistribution remain governed
by each source's terms. Local datasets and caches are excluded from Git.
