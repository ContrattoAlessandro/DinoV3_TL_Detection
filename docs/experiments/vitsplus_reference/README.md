# Preserved ViT-S+ reference release

This directory preserves the release before ViT-B became the default on
8 October 2026. It contains the original fold-0 epoch-5 EMA measurements,
checkpoint metadata, and recorded environment. ViT-S+ remains supported through
[configs/vitsplus.yaml](../../../configs/vitsplus.yaml).

The [original result page](results/README.md) is a historical snapshot: statements
about the default and relative links describe the original repository layout.
The raw DTLD, ATLAS, and VZC reports are preserved byte for byte and checked by
the publication regression tests.

- [DTLD report](results/dtld/report.json)
- [ATLAS report](results/atlas/report.json)
- [VZC report](results/vzc_tld/report.json)
- [Checkpoint identity](checkpoints.json)
- [Recorded environment](environment.json)

Current retained ViT-B measurements are in [docs/results](../../results/README.md).
