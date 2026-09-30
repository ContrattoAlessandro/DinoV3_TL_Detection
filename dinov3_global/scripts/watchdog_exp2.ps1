# Back-compat wrapper for the exp2 (v2) run. Generic logic lives in watchdog.ps1.
& (Join-Path $PSScriptRoot 'watchdog.ps1') -Run 'dinov3_global\runs\exp2' -Config 'dinov3_global\configs\base.yaml' -Epochs 30
