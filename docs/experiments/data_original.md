# Data and annotation protocol

## DTLD

The default data source is DTLD v2.0. Obtain the raw images and annotations
through the [dataset's native parser repository](https://github.com/julimueller/dtld_parsing)
under the applicable data terms. The expected local layout is:

```text
datasets/
  DTLD/
    v2.0/DTLD_train.json
    v2.0/DTLD_test.json
    ... native city/session image directories ...
  DTLD_jpg/train/*.jpg
  DTLD_jpg/test/*.jpg
  DTLD_1280/train/*.jpg
  DTLD_1280/test/*.jpg
```

Run `scripts/preprocessing/convert_dtld.py` to decode the native Bayer TIFFs
through the DTLD parser. It writes RGB JPEGs and skips existing files unless
`--overwrite` is explicitly supplied. `scripts/prepare_dtld.py` crops 114 pixels
from both sides of the original 2048×1024 image, resizes to 1280×720 with bicubic
interpolation, and writes JPEGs at quality 95. The saved image dump is the input
used by the retained training runs.

Annotations remain in the native coordinate system. For an original coordinate
$(x,y)$, the prepared-image coordinate is

$$
x'=\frac{x-114}{2048-228}\,1280,\qquad y'=\frac{y}{1024}\,720.
$$

Boxes are clipped to the valid image region and rasterized onto a 45×80 grid.
The implementation preserves tiny boxes that would otherwise disappear at
patch resolution. Overlapping boxes with conflicting attributes are masked out
of token supervision. A one-patch surrounding ignore band prevents uncertain
box boundaries from becoming background negatives, while preserving all valid
positive tokens.

`crop_sides: 0` means there is no additional input crop of the prepared JPEGs.
`label_crop_sides: 114` records the crop already baked into those images.
Changing one without the other can misalign image pixels and token targets.

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
occur. The prepared train/test dumps have zero shared image identities, shared
sessions, or exact JPEG duplicates. The audit covers 1,478 training sessions and
632 test sessions, with no exact duplicates within either split. Annotation and
ordered membership SHA-256 hashes are recorded in the
[final test protocol](results/dtld_test_protocol.json). The audit hashes the
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
