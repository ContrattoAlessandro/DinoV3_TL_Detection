# Frozen configuration snapshots

These JSON records preserve the values of the original YAML configurations.
They are not supported configuration choices for the current package.

| Snapshot | Experiment |
|:--|:--|
| [default.json](default.json) | Original v5 default |
| [spatial_axial.json](spatial_axial.json) | Axial relevance pilot |
| [generalization_a.json](generalization_a.json) | Full-frame v5 head |
| [generalization_b.json](generalization_b.json) | Revised sampling/supervision |
| [generalization_c.json](generalization_c.json) | Original C pilot and gated selection |
| [generalization_d.json](generalization_d.json) | Higher-resolution C head |

The preservation manifest records both the original YAML hash and JSON snapshot
hash. C's current configuration uses the same model and training losses with
validation-mAP selection; the archived C record retains the original gate.
