# Official Result Watch

This file is auto-generated from `runs/logs/official*.log`.
Use `officialscale` rows for COCO main-table candidates; `officialmap_highres` rows are diagnostics.

## E1 Progress

| method | VOC20 | Context59 | ADE150 | COCO171 | completed | avg mIoU |
|---|---:|---:|---:|---:|---:|---:|
| cass | 87.73 | 40.27 | 20.32 | 26.67 | 4 | 43.75 |
| cliptrase |  |  |  |  | 0 |  |
| corrclip | 88.78 | 48.68 | 26.96 | 32.22 | 4 | 49.16 |
| diffsegmenter |  |  |  |  | 0 |  |
| freeda | 86.19 | 43.42 | 23.19 | 28.80 | 4 | 45.40 |
| naclip | 83.03 | 38.36 | 19.12 | 25.98 | 4 | 41.62 |
| odise |  |  |  |  | 0 |  |
| ovdiff | 64.56 |  |  |  | 1 | 64.56 |
| ovseg |  |  |  |  | 0 |  |
| proxyclip | 80.33 | 39.31 | 19.59 | 26.68 | 4 | 41.48 |
| resclip | 82.31 | 36.85 | 18.07 | 23.59 | 4 | 40.20 |
| san |  |  |  |  | 0 |  |
| scclip | 84.28 | 40.21 | 20.06 | 26.62 | 4 | 42.79 |
| sclip | 81.53 | 34.14 | 16.46 | 22.86 | 4 | 38.75 |
| trident | 83.70 | 40.99 | 20.91 | 27.62 | 4 | 43.30 |

## Candidate Signals

- **e1_partial_average**: Partial E1 average; useful once at least two datasets are complete.
- **e1_partial_average**: Partial E1 average; useful once at least two datasets are complete.
- **e1_partial_average**: Partial E1 average; useful once at least two datasets are complete.
- **e1_partial_average**: Partial E1 average; useful once at least two datasets are complete.
- **e1_partial_average**: Partial E1 average; useful once at least two datasets are complete.
- **e1_partial_average**: Partial E1 average; useful once at least two datasets are complete.
- **e1_partial_average**: Partial E1 average; useful once at least two datasets are complete.
- **e1_partial_average**: Partial E1 average; useful once at least two datasets are complete.
- **e1_partial_average**: Partial E1 average; useful once at least two datasets are complete.
- **dataset_leader**: corrclip leads voc20 by 1.05 mIoU among completed canonical runs.
- **dataset_leader**: corrclip leads context59 by 5.26 mIoU among completed canonical runs.
- **dataset_leader**: corrclip leads ade20k by 3.77 mIoU among completed canonical runs.
- **dataset_leader**: corrclip leads coco_stuff164k by 3.42 mIoU among completed canonical runs.
- **protocol_sensitivity**: naclip/coco_stuff164k changes by 25.43 mIoU across logged protocols.
- **protocol_sensitivity**: naclip/context59 changes by 37.86 mIoU across logged protocols.
- **protocol_sensitivity**: naclip/voc20 changes by 82.90 mIoU across logged protocols.
- **protocol_sensitivity**: proxyclip/voc20 changes by 1.31 mIoU across logged protocols.
- **protocol_sensitivity**: resclip/coco_stuff164k changes by 1.11 mIoU across logged protocols.
- **protocol_sensitivity**: resclip/voc20 changes by 3.80 mIoU across logged protocols.
- **protocol_sensitivity**: scclip/context59 changes by 38.00 mIoU across logged protocols.
- **protocol_sensitivity**: sclip/context59 changes by 33.70 mIoU across logged protocols.
- **protocol_sensitivity**: sclip/voc20 changes by 81.34 mIoU across logged protocols.

## Proposal Novelty Metrics

Rows below are the proposal-listed novelty/exploratory metrics plus a few extra diagnostics.
`computed` rows are derived from current official outputs; `requires_*` rows identify the exact result artifact future official runs must save.

| block | metric | group | status | method | dataset | value | artifact needed |
|---|---|---|---|---|---|---:|---|
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | cass | ade20k | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | cass | ade20k | 48.5600 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | cass | ade20k | 41.5800 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | cass | ade20k | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | cass | ade20k | 2.2049 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | cass | ade20k |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | cass | ade20k | 9.2158 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | cass | ade20k | 0.0333 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | cass | ade20k | 16.6818 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | cass | ade20k | 0.9667 |  |
| extra | artifact completeness | artifact readiness | ready | cass | ade20k | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/cass/ade20k |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | cass | ade847 | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | cass | ade847 | 33.5300 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | cass | ade847 | 20.7200 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | cass | ade847 | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | cass | ade847 | 3.5062 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | cass | ade847 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | cass | ade847 | 2.0507 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | cass | ade847 | 0.4466 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | cass | ade847 | 13.9619 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | cass | ade847 | 0.5534 |  |
| extra | artifact completeness | artifact readiness | ready | cass | ade847 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/cass/ade847 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | cass | coco_stuff164k | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | cass | coco_stuff164k | 43.5600 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | cass | coco_stuff164k | 46.7800 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | cass | coco_stuff164k | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | cass | coco_stuff164k | 1.9929 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | cass | coco_stuff164k |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | cass | coco_stuff164k | 13.3825 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | cass | coco_stuff164k | 0.0058 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | cass | coco_stuff164k | 20.8432 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | cass | coco_stuff164k | 0.9942 |  |
| extra | artifact completeness | artifact readiness | ready | cass | coco_stuff164k | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/cass/coco_stuff164k |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | cass | context459 | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | cass | context459 | 47.8200 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | cass | context459 | 32.2600 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | cass | context459 | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | cass | context459 | 2.1836 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | cass | context459 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | cass | context459 | 3.8377 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | cass | context459 | 0.4323 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | cass | context459 | 15.8133 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | cass | context459 | 0.5677 |  |
| extra | artifact completeness | artifact readiness | ready | cass | context459 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/cass/context459 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | cass | context59 | 3.6500 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | cass | context59 | 64.5800 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | cass | context59 | 61.7900 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | cass | context59 | 3.6500 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | cass | context59 | 2.1003 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | cass | context59 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | cass | context59 | 19.1735 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | cass | context59 | 0.0000 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | cass | context59 | 24.6659 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | cass | context59 | 1.0000 |  |
| extra | artifact completeness | artifact readiness | ready | cass | context59 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/cass/context59 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | cass | voc20 | 40.0300 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | cass | voc20 | 93.8900 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | cass | voc20 | 93.9200 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | cass | voc20 | 40.0300 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | cass | voc20 | 1.2813 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | cass | voc20 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | cass | voc20 | 68.4695 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | cass | voc20 | 0.0000 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | cass | voc20 | 13.6880 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | cass | voc20 | 1.0000 |  |
| extra | artifact completeness | artifact readiness | ready | cass | voc20 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/cass/voc20 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | corrclip | ade20k | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | corrclip | ade20k | 61.7700 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | corrclip | ade20k | 49.4100 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | corrclip | ade20k | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | corrclip | ade20k | 0.0807 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | corrclip | ade20k |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | corrclip | ade20k | 334.0768 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | corrclip | ade20k | 0.0267 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | corrclip | ade20k | 20.6523 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | corrclip | ade20k | 0.9733 |  |
| extra | artifact completeness | artifact readiness | ready | corrclip | ade20k | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/corrclip/ade20k |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | corrclip | ade847 | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | corrclip | ade847 | 48.4700 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | corrclip | ade847 | 20.9300 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | corrclip | ade847 | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | corrclip | ade847 | 0.3358 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | corrclip | ade847 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | corrclip | ade847 | 25.8189 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | corrclip | ade847 | 0.4988 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | corrclip | ade847 | 17.5134 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | corrclip | ade847 | 0.5012 |  |
| extra | artifact completeness | artifact readiness | ready | corrclip | ade847 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/corrclip/ade847 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | corrclip | coco_stuff164k | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | corrclip | coco_stuff164k | 48.6400 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | corrclip | coco_stuff164k | 50.5800 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | corrclip | coco_stuff164k | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | corrclip | coco_stuff164k | 0.0921 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | corrclip | coco_stuff164k |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | corrclip | coco_stuff164k | 349.8371 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | corrclip | coco_stuff164k | 0.0058 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | corrclip | coco_stuff164k | 23.1337 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | corrclip | coco_stuff164k | 0.9942 |  |
| extra | artifact completeness | artifact readiness | ready | corrclip | coco_stuff164k | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/corrclip/coco_stuff164k |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | corrclip | context459 | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | corrclip | context459 | 58.4800 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | corrclip | context459 | 33.8900 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | corrclip | context459 | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | corrclip | context459 | 0.2484 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | corrclip | context459 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | corrclip | context459 | 47.8261 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | corrclip | context459 | 0.4417 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | corrclip | context459 | 20.3982 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | corrclip | context459 | 0.5583 |  |
| extra | artifact completeness | artifact readiness | ready | corrclip | context459 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/corrclip/context459 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | corrclip | context59 | 1.4200 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | corrclip | context59 | 72.6000 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | corrclip | context59 | 69.3900 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | corrclip | context59 | 1.4200 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | corrclip | context59 | 0.0785 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | corrclip | context59 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | corrclip | context59 | 620.1274 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | corrclip | context59 | 0.0000 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | corrclip | context59 | 26.2107 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | corrclip | context59 | 1.0000 |  |
| extra | artifact completeness | artifact readiness | ready | corrclip | context59 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/corrclip/context59 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | corrclip | voc20 | 49.7100 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | corrclip | voc20 | 94.5300 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | corrclip | voc20 | 94.5000 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | corrclip | voc20 | 49.7100 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | corrclip | voc20 | 0.0758 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | corrclip | voc20 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | corrclip | voc20 | 1171.2401 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | corrclip | voc20 | 0.0000 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | corrclip | voc20 | 11.6877 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | corrclip | voc20 | 1.0000 |  |
| extra | artifact completeness | artifact readiness | ready | corrclip | voc20 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/corrclip/voc20 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | freeda | ade20k | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | freeda | ade20k | 48.3700 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | freeda | ade20k | 47.9600 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | freeda | ade20k | 0.0000 |  |
| E4 | inference time | standard cost reporting | pending_e4_log | freeda | ade20k |  | official timing log |
| E4 | peak memory | standard cost reporting | pending_memory_log | freeda | ade20k |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | pending_e4_log | freeda | ade20k |  | official timing log |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | freeda | ade20k | 0.0133 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | freeda | ade20k | 19.3911 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | freeda | ade20k | 0.9867 |  |
| extra | artifact completeness | artifact readiness | ready | freeda | ade20k | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/freeda/ade20k |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | freeda | ade847 | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | freeda | ade847 | 28.0800 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | freeda | ade847 | 22.8500 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | freeda | ade847 | 0.0000 |  |
| E4 | inference time | standard cost reporting | pending_e4_log | freeda | ade847 |  | official timing log |
| E4 | peak memory | standard cost reporting | pending_memory_log | freeda | ade847 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | pending_e4_log | freeda | ade847 |  | official timing log |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | freeda | ade847 | 0.7266 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | freeda | ade847 | 10.0996 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | freeda | ade847 | 0.2734 |  |
| extra | artifact completeness | artifact readiness | ready | freeda | ade847 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/freeda/ade847 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | freeda | coco_stuff164k | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | freeda | coco_stuff164k | 44.3100 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | freeda | coco_stuff164k | 52.6100 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | freeda | coco_stuff164k | 0.0000 |  |
| E4 | inference time | standard cost reporting | pending_e4_log | freeda | coco_stuff164k |  | official timing log |
| E4 | peak memory | standard cost reporting | pending_memory_log | freeda | coco_stuff164k |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | pending_e4_log | freeda | coco_stuff164k |  | official timing log |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | freeda | coco_stuff164k | 0.0117 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | freeda | coco_stuff164k | 22.3561 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | freeda | coco_stuff164k | 0.9883 |  |
| extra | artifact completeness | artifact readiness | ready | freeda | coco_stuff164k | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/freeda/coco_stuff164k |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | freeda | context459 | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | freeda | context459 | 21.3300 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | freeda | context459 | 20.1400 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | freeda | context459 | 0.0000 |  |
| E4 | inference time | standard cost reporting | pending_e4_log | freeda | context459 |  | official timing log |
| E4 | peak memory | standard cost reporting | pending_memory_log | freeda | context459 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | pending_e4_log | freeda | context459 |  | official timing log |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | freeda | context459 | 0.6841 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | freeda | context459 | 12.1559 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | freeda | context459 | 0.3159 |  |
| extra | artifact completeness | artifact readiness | ready | freeda | context459 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/freeda/context459 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | freeda | context59 | 2.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | freeda | context59 | 66.0300 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | freeda | context59 | 64.6000 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | freeda | context59 | 2.0000 |  |
| E4 | inference time | standard cost reporting | pending_e4_log | freeda | context59 |  | official timing log |
| E4 | peak memory | standard cost reporting | pending_memory_log | freeda | context59 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | pending_e4_log | freeda | context59 |  | official timing log |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | freeda | context59 | 0.0000 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | freeda | context59 | 24.1604 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | freeda | context59 | 1.0000 |  |
| extra | artifact completeness | artifact readiness | ready | freeda | context59 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/freeda/context59 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | freeda | voc20 | 37.3400 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | freeda | voc20 | 93.0100 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | freeda | voc20 | 93.3200 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | freeda | voc20 | 37.3400 |  |
| E4 | inference time | standard cost reporting | pending_e4_log | freeda | voc20 |  | official timing log |
| E4 | peak memory | standard cost reporting | pending_memory_log | freeda | voc20 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | pending_e4_log | freeda | voc20 |  | official timing log |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | freeda | voc20 | 0.0000 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | freeda | voc20 | 13.9018 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | freeda | voc20 | 1.0000 |  |
| extra | artifact completeness | artifact readiness | ready | freeda | voc20 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/freeda/voc20 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | naclip | ade20k | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | naclip | ade20k | 48.1900 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | naclip | ade20k | 39.3500 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | naclip | ade20k | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | naclip | ade20k | 0.4780 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | naclip | ade20k |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | naclip | ade20k | 40.0000 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | naclip | ade20k | 0.0467 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | naclip | ade20k | 15.7897 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | naclip | ade20k | 0.9533 |  |
| extra | artifact completeness | artifact readiness | ready | naclip | ade20k | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/naclip/ade20k |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | naclip | ade847 | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | naclip | ade847 | 33.8500 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | naclip | ade847 | 20.1600 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | naclip | ade847 | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | naclip | ade847 | 1.9217 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | naclip | ade847 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | naclip | ade847 | 3.1274 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | naclip | ade847 | 0.3625 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | naclip | ade847 | 11.0988 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | naclip | ade847 | 0.6375 |  |
| extra | artifact completeness | artifact readiness | ready | naclip | ade847 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/naclip/ade847 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | naclip | coco_stuff164k | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | naclip | coco_stuff164k | 41.6300 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | naclip | coco_stuff164k | 44.9400 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | naclip | coco_stuff164k | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | naclip | coco_stuff164k | 0.4071 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | naclip | coco_stuff164k |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | naclip | coco_stuff164k | 63.8172 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | naclip | coco_stuff164k | 0.0058 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | naclip | coco_stuff164k | 20.4058 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | naclip | coco_stuff164k | 0.9942 |  |
| extra | artifact completeness | artifact readiness | ready | naclip | coco_stuff164k | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/naclip/coco_stuff164k |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | naclip | context459 | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | naclip | context459 | 47.0300 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | naclip | context459 | 29.4300 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | naclip | context459 | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | naclip | context459 | 0.8143 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | naclip | context459 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | naclip | context459 | 9.5665 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | naclip | context459 | 0.3856 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | naclip | context459 | 14.6105 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | naclip | context459 | 0.6144 |  |
| extra | artifact completeness | artifact readiness | ready | naclip | context459 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/naclip/context459 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | naclip | context59 | 4.0600 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | naclip | context59 | 62.2800 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | naclip | context59 | 59.1300 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | naclip | context59 | 4.0600 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | naclip | context59 | 0.1373 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | naclip | context59 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | naclip | context59 | 279.3882 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | naclip | context59 | 0.0000 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | naclip | context59 | 24.3235 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | naclip | context59 | 1.0000 |  |
| extra | artifact completeness | artifact readiness | ready | naclip | context59 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/naclip/context59 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | naclip | voc20 | 47.1800 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | naclip | voc20 | 91.2500 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | naclip | voc20 | 90.7900 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | naclip | voc20 | 47.1800 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | naclip | voc20 | 0.0956 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | naclip | voc20 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | naclip | voc20 | 868.5146 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | naclip | voc20 | 0.0000 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | naclip | voc20 | 11.7746 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | naclip | voc20 | 1.0000 |  |
| extra | artifact completeness | artifact readiness | ready | naclip | voc20 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/naclip/voc20 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | ovdiff | voc20 | 22.9038 |  |
| E1 | pixel accuracy | Group (i); cheap | pending_log_summary | ovdiff | voc20 |  | official mmseg aAcc |
| E1 | mean class accuracy | Group (i); cheap | pending_log_summary | ovdiff | voc20 |  | official mmseg mAcc |
| E3 | worst-class within target | Group (i) | computed_from_official_log | ovdiff | voc20 | 22.9038 |  |
| E4 | inference time | standard cost reporting | pending_e4_log | ovdiff | voc20 |  | official timing log |
| E4 | peak memory | standard cost reporting | pending_memory_log | ovdiff | voc20 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | pending_e4_log | ovdiff | voc20 |  | official timing log |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | ovdiff | voc20 | 0.0000 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | ovdiff | voc20 | 20.6071 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | ovdiff | voc20 | 1.0000 |  |
| extra | artifact completeness | artifact readiness | missing | ovdiff | voc20 | 0.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/ovdiff/voc20 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | proxyclip | ade20k | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | proxyclip | ade20k | 46.8000 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | proxyclip | ade20k | 41.1800 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | proxyclip | ade20k | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | proxyclip | ade20k | 0.0568 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | proxyclip | ade20k |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | proxyclip | ade20k | 344.8944 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | proxyclip | ade20k | 0.0533 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | proxyclip | ade20k | 16.3392 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | proxyclip | ade20k | 0.9467 |  |
| extra | artifact completeness | artifact readiness | ready | proxyclip | ade20k | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/proxyclip/ade20k |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | proxyclip | ade847 | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | proxyclip | ade847 | 34.8600 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | proxyclip | ade847 | 21.2300 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | proxyclip | ade847 | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | proxyclip | ade847 | 0.3546 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | proxyclip | ade847 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | proxyclip | ade847 | 19.4585 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | proxyclip | ade847 | 0.4202 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | proxyclip | ade847 | 13.4951 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | proxyclip | ade847 | 0.5798 |  |
| extra | artifact completeness | artifact readiness | ready | proxyclip | ade847 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/proxyclip/ade847 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | proxyclip | coco_stuff164k | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | proxyclip | coco_stuff164k | 42.9600 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | proxyclip | coco_stuff164k | 47.5500 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | proxyclip | coco_stuff164k | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | proxyclip | coco_stuff164k | 0.0753 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | proxyclip | coco_stuff164k |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | proxyclip | coco_stuff164k | 354.3161 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | proxyclip | coco_stuff164k | 0.0117 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | proxyclip | coco_stuff164k | 20.7942 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | proxyclip | coco_stuff164k | 0.9883 |  |
| extra | artifact completeness | artifact readiness | ready | proxyclip | coco_stuff164k | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/proxyclip/coco_stuff164k |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | proxyclip | context459 | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | proxyclip | context459 | 48.1500 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | proxyclip | context459 | 34.2500 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | proxyclip | context459 | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | proxyclip | context459 | 0.3974 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | proxyclip | context459 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | proxyclip | context459 | 21.1626 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | proxyclip | context459 | 0.4236 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | proxyclip | context459 | 15.5453 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | proxyclip | context459 | 0.5764 |  |
| extra | artifact completeness | artifact readiness | ready | proxyclip | context459 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/proxyclip/context459 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | proxyclip | context59 | 3.6200 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | proxyclip | context59 | 62.8500 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | proxyclip | context59 | 61.8100 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | proxyclip | context59 | 3.6200 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | proxyclip | context59 | 0.0658 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | proxyclip | context59 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | proxyclip | context59 | 597.4164 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | proxyclip | context59 | 0.0000 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | proxyclip | context59 | 24.4897 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | proxyclip | context59 | 1.0000 |  |
| extra | artifact completeness | artifact readiness | ready | proxyclip | context59 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/proxyclip/context59 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | proxyclip | voc20 | 41.7600 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | proxyclip | voc20 | 89.0600 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | proxyclip | voc20 | 90.0000 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | proxyclip | voc20 | 41.7600 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | proxyclip | voc20 | 0.0569 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | proxyclip | voc20 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | proxyclip | voc20 | 1411.7750 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | proxyclip | voc20 | 0.0000 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | proxyclip | voc20 | 14.7833 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | proxyclip | voc20 | 1.0000 |  |
| extra | artifact completeness | artifact readiness | ready | proxyclip | voc20 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/proxyclip/voc20 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | resclip | ade20k | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | resclip | ade20k | 45.1300 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | resclip | ade20k | 40.3300 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | resclip | ade20k | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | resclip | ade20k | 0.2791 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | resclip | ade20k |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | resclip | ade20k | 64.7438 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | resclip | ade20k | 0.0467 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | resclip | ade20k | 14.9687 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | resclip | ade20k | 0.9533 |  |
| extra | artifact completeness | artifact readiness | ready | resclip | ade20k | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/resclip/ade20k |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | resclip | ade847 | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | resclip | ade847 | 33.6600 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | resclip | ade847 | 21.4300 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | resclip | ade847 | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | resclip | ade847 | 4.4533 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | resclip | ade847 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | resclip | ade847 | 1.4573 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | resclip | ade847 | 0.3915 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | resclip | ade847 | 12.2932 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | resclip | ade847 | 0.6085 |  |
| extra | artifact completeness | artifact readiness | ready | resclip | ade847 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/resclip/ade847 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | resclip | coco_stuff164k | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | resclip | coco_stuff164k | 40.2000 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | resclip | coco_stuff164k | 44.0000 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | resclip | coco_stuff164k | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | resclip | coco_stuff164k | 0.3497 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | resclip | coco_stuff164k |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | resclip | coco_stuff164k | 67.4578 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | resclip | coco_stuff164k | 0.0117 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | resclip | coco_stuff164k | 18.4596 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | resclip | coco_stuff164k | 0.9883 |  |
| extra | artifact completeness | artifact readiness | ready | resclip | coco_stuff164k | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/resclip/coco_stuff164k |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | resclip | context459 | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | resclip | context459 | 45.3300 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | resclip | context459 | 32.7900 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | resclip | context459 | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | resclip | context459 | 2.2704 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | resclip | context459 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | resclip | context459 | 3.3915 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | resclip | context459 | 0.3943 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | resclip | context459 | 14.2298 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | resclip | context459 | 0.6057 |  |
| extra | artifact completeness | artifact readiness | ready | resclip | context459 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/resclip/context459 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | resclip | context59 | 5.4200 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | resclip | context59 | 60.3100 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | resclip | context59 | 60.2600 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | resclip | context59 | 5.4200 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | resclip | context59 | 0.1616 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | resclip | context59 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | resclip | context59 | 228.0322 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | resclip | context59 | 0.0000 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | resclip | context59 | 22.8836 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | resclip | context59 | 1.0000 |  |
| extra | artifact completeness | artifact readiness | ready | resclip | context59 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/resclip/context59 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | resclip | voc20 | 41.7700 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | resclip | voc20 | 91.4900 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | resclip | voc20 | 90.9900 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | resclip | voc20 | 41.7700 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | resclip | voc20 | 0.1191 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | resclip | voc20 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | resclip | voc20 | 691.0999 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | resclip | voc20 | 0.0000 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | resclip | voc20 | 12.7631 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | resclip | voc20 | 1.0000 |  |
| extra | artifact completeness | artifact readiness | ready | resclip | voc20 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/resclip/voc20 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | scclip | ade20k | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | scclip | ade20k | 49.8700 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | scclip | ade20k | 41.7700 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | scclip | ade20k | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | scclip | ade20k | 0.1486 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | scclip | ade20k |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | scclip | ade20k | 134.9933 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | scclip | ade20k | 0.0467 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | scclip | ade20k | 16.4196 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | scclip | ade20k | 0.9533 |  |
| extra | artifact completeness | artifact readiness | ready | scclip | ade20k | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/scclip/ade20k |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | scclip | ade847 | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | scclip | ade847 | 38.1200 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | scclip | ade847 | 22.1000 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | scclip | ade847 | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | scclip | ade847 | 0.1862 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | scclip | ade847 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | scclip | ade847 | 40.0644 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | scclip | ade847 | 0.4076 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | scclip | ade847 | 13.9699 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | scclip | ade847 | 0.5924 |  |
| extra | artifact completeness | artifact readiness | ready | scclip | ade847 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/scclip/ade847 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | scclip | coco_stuff164k | 0.0100 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | scclip | coco_stuff164k | 45.3600 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | scclip | coco_stuff164k | 48.3800 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | scclip | coco_stuff164k | 0.0100 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | scclip | coco_stuff164k | 0.1686 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | scclip | coco_stuff164k |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | scclip | coco_stuff164k | 157.8885 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | scclip | coco_stuff164k | 0.0000 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | scclip | coco_stuff164k | 20.4971 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | scclip | coco_stuff164k | 1.0000 |  |
| extra | artifact completeness | artifact readiness | ready | scclip | coco_stuff164k | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/scclip/coco_stuff164k |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | scclip | context459 | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | scclip | context459 | 49.5000 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | scclip | context459 | 33.1300 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | scclip | context459 | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | scclip | context459 | 0.0865 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | scclip | context459 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | scclip | context459 | 96.4162 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | scclip | context459 | 0.4148 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | scclip | context459 | 15.4750 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | scclip | context459 | 0.5852 |  |
| extra | artifact completeness | artifact readiness | ready | scclip | context459 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/scclip/context459 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | scclip | context59 | 4.0500 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | scclip | context59 | 65.0200 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | scclip | context59 | 62.6500 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | scclip | context59 | 4.0500 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | scclip | context59 | 0.1506 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | scclip | context59 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | scclip | context59 | 266.9987 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | scclip | context59 | 0.0000 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | scclip | context59 | 23.7508 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | scclip | context59 | 1.0000 |  |
| extra | artifact completeness | artifact readiness | ready | scclip | context59 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/scclip/context59 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | scclip | voc20 | 42.9800 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | scclip | voc20 | 92.5500 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | scclip | voc20 | 92.0100 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | scclip | voc20 | 42.9800 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | scclip | voc20 | 0.1221 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | scclip | voc20 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | scclip | voc20 | 690.2539 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | scclip | voc20 | 0.0000 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | scclip | voc20 | 11.8522 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | scclip | voc20 | 1.0000 |  |
| extra | artifact completeness | artifact readiness | ready | scclip | voc20 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/scclip/voc20 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | sclip | ade20k | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | sclip | ade20k | 38.7200 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | sclip | ade20k | 37.0800 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | sclip | ade20k | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | sclip | ade20k | 0.0512 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | sclip | ade20k |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | sclip | ade20k | 321.4844 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | sclip | ade20k | 0.0333 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | sclip | ade20k | 14.3746 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | sclip | ade20k | 0.9667 |  |
| extra | artifact completeness | artifact readiness | ready | sclip | ade20k | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/sclip/ade20k |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | sclip | ade847 | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | sclip | ade847 | 28.3300 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | sclip | ade847 | 19.7200 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | sclip | ade847 | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | sclip | ade847 | 0.2903 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | sclip | ade847 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | sclip | ade847 | 19.5315 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | sclip | ade847 | 0.3917 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | sclip | ade847 | 11.3064 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | sclip | ade847 | 0.6083 |  |
| extra | artifact completeness | artifact readiness | ready | sclip | ade847 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/sclip/ade847 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | sclip | coco_stuff164k | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | sclip | coco_stuff164k | 38.3300 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | sclip | coco_stuff164k | 41.7700 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | sclip | coco_stuff164k | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | sclip | coco_stuff164k | 0.1119 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | sclip | coco_stuff164k |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | sclip | coco_stuff164k | 204.2895 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | sclip | coco_stuff164k | 0.0058 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | sclip | coco_stuff164k | 18.9919 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | sclip | coco_stuff164k | 0.9942 |  |
| extra | artifact completeness | artifact readiness | ready | sclip | coco_stuff164k | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/sclip/coco_stuff164k |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | sclip | context459 | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | sclip | context459 | 43.1800 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | sclip | context459 | 30.1400 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | sclip | context459 | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | sclip | context459 | 0.1370 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | sclip | context459 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | sclip | context459 | 48.7591 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | sclip | context459 | 0.3965 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | sclip | context459 | 13.4848 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | sclip | context459 | 0.6035 |  |
| extra | artifact completeness | artifact readiness | ready | sclip | context459 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/sclip/context459 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | sclip | context59 | 2.1600 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | sclip | context59 | 57.7300 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | sclip | context59 | 55.7200 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | sclip | context59 | 2.1600 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | sclip | context59 | 0.0513 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | sclip | context59 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | sclip | context59 | 665.4971 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | sclip | context59 | 0.0000 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | sclip | context59 | 22.7864 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | sclip | context59 | 1.0000 |  |
| extra | artifact completeness | artifact readiness | ready | sclip | context59 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/sclip/context59 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | sclip | voc20 | 42.6100 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | sclip | voc20 | 91.0200 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | sclip | voc20 | 90.2900 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | sclip | voc20 | 42.6100 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | sclip | voc20 | 0.0506 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | sclip | voc20 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | sclip | voc20 | 1611.2648 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | sclip | voc20 | 0.0000 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | sclip | voc20 | 13.0180 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | sclip | voc20 | 1.0000 |  |
| extra | artifact completeness | artifact readiness | ready | sclip | voc20 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/sclip/voc20 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | trident | ade20k | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | trident | ade20k | 50.5700 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | trident | ade20k | 42.4900 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | trident | ade20k | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | trident | ade20k | 0.0736 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | trident | ade20k |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | trident | ade20k | 284.1033 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | trident | ade20k | 0.0467 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | trident | ade20k | 17.1094 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | trident | ade20k | 0.9533 |  |
| extra | artifact completeness | artifact readiness | ready | trident | ade20k | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/trident/ade20k |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | trident | ade847 | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | trident | ade847 | 34.4000 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | trident | ade847 | 21.3800 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | trident | ade847 | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | trident | ade847 | 0.1524 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | trident | ade847 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | trident | ade847 | 48.8845 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | trident | ade847 | 0.4705 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | trident | ade847 | 14.8614 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | trident | ade847 | 0.5295 |  |
| extra | artifact completeness | artifact readiness | ready | trident | ade847 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/trident/ade847 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | trident | coco_stuff164k | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | trident | coco_stuff164k | 44.8300 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | trident | coco_stuff164k | 48.7400 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | trident | coco_stuff164k | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | trident | coco_stuff164k | 0.0688 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | trident | coco_stuff164k |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | trident | coco_stuff164k | 401.4535 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | trident | coco_stuff164k | 0.0117 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | trident | coco_stuff164k | 21.6210 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | trident | coco_stuff164k | 0.9883 |  |
| extra | artifact completeness | artifact readiness | ready | trident | coco_stuff164k | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/trident/coco_stuff164k |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | trident | context459 | 0.0000 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | trident | context459 | 49.5700 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | trident | context459 | 34.8200 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | trident | context459 | 0.0000 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | trident | context459 | 0.2159 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | trident | context459 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | trident | context459 | 41.5470 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | trident | context459 | 0.4420 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | trident | context459 | 16.2788 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | trident | context459 | 0.5580 |  |
| extra | artifact completeness | artifact readiness | ready | trident | context459 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/trident/context459 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | trident | context59 | 2.6100 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | trident | context59 | 64.9600 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | trident | context59 | 63.5700 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | trident | context59 | 2.6100 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | trident | context59 | 0.0716 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | trident | context59 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | trident | context59 | 572.4860 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | trident | context59 | 0.0000 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | trident | context59 | 24.5808 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | trident | context59 | 1.0000 |  |
| extra | artifact completeness | artifact readiness | ready | trident | context59 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/trident/context59 |
| E1 | per-class IoU summary | Group (i); cheap | computed_from_official_log | trident | voc20 | 42.6200 |  |
| E1 | pixel accuracy | Group (i); cheap | computed_from_official_log | trident | voc20 | 90.6100 |  |
| E1 | mean class accuracy | Group (i); cheap | computed_from_official_log | trident | voc20 | 92.1800 |  |
| E3 | worst-class within target | Group (i) | computed_from_official_log | trident | voc20 | 42.6200 |  |
| E4 | inference time | standard cost reporting | computed_partial_from_official_log | trident | voc20 | 0.0612 |  |
| E4 | peak memory | standard cost reporting | pending_memory_log | trident | voc20 |  | official memory field or nvidia-smi sampler |
| E4 | mIoU per second | Group (ii); novelty-supporting | computed_partial_from_official_log | trident | voc20 | 1367.6471 |  |
| extra | zero-IoU class collapse rate | extra discovery | computed_from_official_log | trident | voc20 | 0.0000 |  |
| extra | class-IoU dispersion | extra discovery | computed_from_official_log | trident | voc20 | 15.7202 |  |
| extra | nonzero-class coverage | extra discovery | computed_from_official_log | trident | voc20 | 1.0000 |  |
| extra | artifact completeness | artifact readiness | ready | trident | voc20 | 100.0000 | /data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/trident/voc20 |
| E2 | Delta_vocab_pair | Group (ii); novelty-driving | computed | cass | context59->context459 | 31.8900 |  |
| E2 | Delta_vocab_pair | Group (ii); novelty-driving | computed | cass | ade20k->ade847 | 13.1300 |  |
| E2 | Delta_vocab | Group (ii); novelty-driving | computed | cass | context+ade | 22.5100 |  |
| E2 | Delta_vocab_pair | Group (ii); novelty-driving | computed | corrclip | context59->context459 | 36.8000 |  |
| E2 | Delta_vocab_pair | Group (ii); novelty-driving | computed | corrclip | ade20k->ade847 | 18.2900 |  |
| E2 | Delta_vocab | Group (ii); novelty-driving | computed | corrclip | context+ade | 27.5450 |  |
| E2 | Delta_vocab_pair | Group (ii); novelty-driving | computed | freeda | context59->context459 | 39.1500 |  |
| E2 | Delta_vocab_pair | Group (ii); novelty-driving | computed | freeda | ade20k->ade847 | 19.2600 |  |
| E2 | Delta_vocab | Group (ii); novelty-driving | computed | freeda | context+ade | 29.2050 |  |
| E2 | Delta_vocab_pair | Group (ii); novelty-driving | computed | naclip | context59->context459 | 30.5700 |  |
| E2 | Delta_vocab_pair | Group (ii); novelty-driving | computed | naclip | ade20k->ade847 | 13.1100 |  |
| E2 | Delta_vocab | Group (ii); novelty-driving | computed | naclip | context+ade | 21.8400 |  |
| E2 | Delta_vocab_pair | Group (ii); novelty-driving | computed | proxyclip | context59->context459 | 30.9000 |  |
| E2 | Delta_vocab_pair | Group (ii); novelty-driving | computed | proxyclip | ade20k->ade847 | 12.6900 |  |
| E2 | Delta_vocab | Group (ii); novelty-driving | computed | proxyclip | context+ade | 21.7950 |  |
| E2 | Delta_vocab_pair | Group (ii); novelty-driving | computed | resclip | context59->context459 | 29.1500 |  |
| E2 | Delta_vocab_pair | Group (ii); novelty-driving | computed | resclip | ade20k->ade847 | 11.5800 |  |
| E2 | Delta_vocab | Group (ii); novelty-driving | computed | resclip | context+ade | 20.3650 |  |
| E2 | Delta_vocab_pair | Group (ii); novelty-driving | computed | scclip | context59->context459 | 31.8700 |  |
| E2 | Delta_vocab_pair | Group (ii); novelty-driving | computed | scclip | ade20k->ade847 | 12.6000 |  |
| E2 | Delta_vocab | Group (ii); novelty-driving | computed | scclip | context+ade | 22.2350 |  |
| E2 | Delta_vocab_pair | Group (ii); novelty-driving | computed | sclip | context59->context459 | 27.4600 |  |
| E2 | Delta_vocab_pair | Group (ii); novelty-driving | computed | sclip | ade20k->ade847 | 10.7900 |  |
| E2 | Delta_vocab | Group (ii); novelty-driving | computed | sclip | context+ade | 19.1250 |  |
| E2 | Delta_vocab_pair | Group (ii); novelty-driving | computed | trident | context59->context459 | 32.0200 |  |
| E2 | Delta_vocab_pair | Group (ii); novelty-driving | computed | trident | ade20k->ade847 | 13.4600 |  |
| E2 | Delta_vocab | Group (ii); novelty-driving | computed | trident | context+ade | 22.7400 |  |
| E3 | Spearman rank correlation | Group (ii); novelty-driving | computed | all | voc20<->context59 | 0.8167 |  |
| E3 | Kendall tau rank correlation | Group (ii); novelty-driving | computed | all | voc20<->context59 | 0.6667 |  |
| E3 | Spearman rank correlation | Group (ii); novelty-driving | computed | all | voc20<->ade20k | 0.8167 |  |
| E3 | Kendall tau rank correlation | Group (ii); novelty-driving | computed | all | voc20<->ade20k | 0.6667 |  |
| E3 | Spearman rank correlation | Group (ii); novelty-driving | computed | all | voc20<->coco_stuff164k | 0.6167 |  |
| E3 | Kendall tau rank correlation | Group (ii); novelty-driving | computed | all | voc20<->coco_stuff164k | 0.5556 |  |
| E3 | Spearman rank correlation | Group (ii); novelty-driving | computed | all | context59<->ade20k | 1.0000 |  |
| E3 | Kendall tau rank correlation | Group (ii); novelty-driving | computed | all | context59<->ade20k | 1.0000 |  |
| E3 | Spearman rank correlation | Group (ii); novelty-driving | computed | all | context59<->coco_stuff164k | 0.9500 |  |
| E3 | Kendall tau rank correlation | Group (ii); novelty-driving | computed | all | context59<->coco_stuff164k | 0.8889 |  |
| E3 | Spearman rank correlation | Group (ii); novelty-driving | computed | all | ade20k<->coco_stuff164k | 0.9500 |  |
| E3 | Kendall tau rank correlation | Group (ii); novelty-driving | computed | all | ade20k<->coco_stuff164k | 0.8889 |  |
| E3 | worst-target mIoU | Group (i) | computed | cass | ade20k | 20.3200 |  |
| E3 | worst-target mIoU | Group (i) | computed | corrclip | ade20k | 26.9600 |  |
| E3 | worst-target mIoU | Group (i) | computed | freeda | ade20k | 23.1900 |  |
| E3 | worst-target mIoU | Group (i) | computed | naclip | ade20k | 19.1200 |  |
| E3 | worst-target mIoU | Group (i) | partial_one_target | ovdiff | voc20 | 64.5616 |  |
| E3 | worst-target mIoU | Group (i) | computed | proxyclip | ade20k | 19.5900 |  |
| E3 | worst-target mIoU | Group (i) | computed | resclip | ade20k | 18.0700 |  |
| E3 | worst-target mIoU | Group (i) | computed | scclip | ade20k | 20.0600 |  |
| E3 | worst-target mIoU | Group (i) | computed | sclip | ade20k | 16.4600 |  |
| E3 | worst-target mIoU | Group (i) | computed | trident | ade20k | 20.9100 |  |
| extra | protocol / scale sensitivity | extra discovery | computed | corrclip | voc20 | 0.2300 |  |
| extra | protocol / scale sensitivity | extra discovery | computed | naclip | ade20k | 0.0700 |  |
| extra | protocol / scale sensitivity | extra discovery | computed | naclip | coco_stuff164k | 25.4300 |  |
| extra | protocol / scale sensitivity | extra discovery | computed | naclip | context59 | 37.8600 |  |
| extra | protocol / scale sensitivity | extra discovery | computed | naclip | voc20 | 82.9000 |  |
| extra | protocol / scale sensitivity | extra discovery | computed | proxyclip | ade20k | 0.1200 |  |
| extra | protocol / scale sensitivity | extra discovery | computed | proxyclip | coco_stuff164k | 0.1500 |  |
| extra | protocol / scale sensitivity | extra discovery | computed | proxyclip | context59 | 0.0000 |  |
| extra | protocol / scale sensitivity | extra discovery | computed | proxyclip | voc20 | 1.3100 |  |
| extra | protocol / scale sensitivity | extra discovery | computed | resclip | coco_stuff164k | 1.1100 |  |
| extra | protocol / scale sensitivity | extra discovery | computed | resclip | voc20 | 3.8000 |  |
| extra | protocol / scale sensitivity | extra discovery | computed | scclip | context59 | 38.0000 |  |
| extra | protocol / scale sensitivity | extra discovery | computed | sclip | ade20k | 0.0000 |  |
| extra | protocol / scale sensitivity | extra discovery | computed | sclip | coco_stuff164k | 0.0900 |  |
| extra | protocol / scale sensitivity | extra discovery | computed | sclip | context59 | 33.7000 |  |
| extra | protocol / scale sensitivity | extra discovery | computed | sclip | voc20 | 81.3400 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | cass | voc20 | 1 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | corrclip | voc20 | 0 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | naclip | voc20 | 1 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | proxyclip | voc20 | 1 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | resclip | voc20 | 1 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | scclip | voc20 | 1 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | sclip | voc20 | 0 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | trident | voc20 | 0 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | cass | context59 | 1 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | corrclip | context59 | 0 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | naclip | context59 | 1 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | proxyclip | context59 | 0 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | resclip | context59 | 1 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | scclip | context59 | 1 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | sclip | context59 | 0 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | trident | context59 | 0 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | cass | ade20k | 1 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | corrclip | ade20k | 0 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | naclip | ade20k | 1 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | proxyclip | ade20k | 0 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | resclip | ade20k | 1 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | scclip | ade20k | 1 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | sclip | ade20k | 0 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | trident | ade20k | 0 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | cass | coco_stuff164k | 1 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | corrclip | coco_stuff164k | 0 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | naclip | coco_stuff164k | 1 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | proxyclip | coco_stuff164k | 1 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | resclip | coco_stuff164k | 1 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | scclip | coco_stuff164k | 1 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | sclip | coco_stuff164k | 1 |  |
| extra | Pareto dominance flag | extra discovery | computed_latency_only | trident | coco_stuff164k | 0 |  |
| extra | method-family gap | extra discovery | computed | family_average | voc20 | 9.7592 |  |
| extra | method-family gap | extra discovery | computed | family_average | context59 | 6.0300 |  |
| extra | method-family gap | extra discovery | computed | family_average | ade20k | 4.7625 |  |
| extra | method-family gap | extra discovery | computed | family_average | coco_stuff164k | 4.0375 |  |
| E1 | thing/stuff mIoU | Group (i); cheap | requires_taxonomy | all |  |  | per-class IoU plus dataset thing/stuff taxonomy |
| E1 | frequency-stratified mIoU | Group (i); cheap | requires_taxonomy | all |  |  | per-class IoU plus train-frequency bins |
| E1 | small/medium/large object mIoU | Group (i); cheap | requires_prediction_artifacts | all |  |  | prediction label maps plus GT instance/region areas |
| E1 | BIoU | Group (i); cheap | requires_prediction_artifacts | all |  |  | prediction label maps plus GT boundary maps |
| E1 | proposal Recall@0.5/0.7 | Group (i); cheap | requires_prediction_artifacts | all |  |  | class-agnostic proposal masks plus GT masks |
| E1 | confidence calibration ECE | Group (i); cheap | requires_score_artifacts | all |  |  | per-pixel or per-region confidence and correctness |
| E1 | confidence calibration Brier | Group (i); cheap | requires_score_artifacts | all |  |  | per-pixel or per-region confidence and correctness |
| E1 | empty-prediction rate | Group (i); cheap | requires_prediction_artifacts | all |  |  | prediction label maps |
| E1 | over-segmentation rate | Group (i); cheap | requires_prediction_artifacts | all |  |  | prediction label maps or proposal masks |
| E1 | prediction coverage | Group (i); cheap | requires_prediction_artifacts | all |  |  | prediction label maps |
| E2 | MCMR@0.5 | Group (ii); novelty-driving | requires_region_artifacts | all |  |  | matched predicted regions, GT regions, predicted labels |
| E2 | MCMR@0.75 | Group (ii); novelty-driving | requires_region_artifacts | all |  |  | matched predicted regions, GT regions, predicted labels |
| E2 | semantic-distance-weighted mismatch | Group (ii); novelty-driving | requires_region_artifacts | all |  |  | mismatch pairs plus CLIP/WordNet label distances |
| E2 | hierarchical IoU | Group (ii); novelty-driving | requires_taxonomy | all |  |  | prediction/GT labels plus WordNet or dataset hierarchy |
| E2 | synonym-collapsed mIoU | Group (ii); novelty-driving | requires_taxonomy | all |  |  | prediction/GT labels plus synonym mapping |
| E2 | prompt-sensitivity variance | Group (ii); novelty-driving | requires_controlled_runs | all |  |  | fixed prompt-template reruns per method |
| E2 | vocabulary-size scaling curve | Group (ii); novelty-driving | requires_controlled_runs | all |  |  | subsampled Context/ADE vocabulary reruns |
| E2 | region-level Top-1 naming accuracy | Group (iii); probe | requires_score_artifacts | all |  |  | top-k class scores for best-matched regions |
| E2 | region-level Top-3 naming accuracy | Group (iii); probe | requires_score_artifacts | all |  |  | top-k class scores for best-matched regions |
| E2 | region-level Top-5 naming accuracy | Group (iii); probe | requires_score_artifacts | all |  |  | top-k class scores for best-matched regions |
| E2 | GT-region naming Top-1 | Group (iii); probe | requires_probe_outputs | all |  |  | GT-crop naming probe outputs |
| E2 | GT-region naming Top-5 | Group (iii); probe | requires_probe_outputs | all |  |  | GT-crop naming probe outputs |
| E2 | GT-text localization IoU | Group (iii); probe | requires_probe_outputs | all |  |  | GT-class prompt localization outputs |
| E2 | GT-text localization BIoU | Group (iii); probe | requires_probe_outputs | all |  |  | GT-class prompt localization outputs |
| E2 | proposal-recall oracle BestIoU | Group (iii); probe | requires_region_artifacts | all |  |  | class-agnostic proposals plus GT regions |
| E2 | MCC consistency before/after | Group (iii); probe | requires_controlled_runs | all |  |  | baseline and post-hoc mask-category re-ranking outputs |
| E2 | top mismatch pairs | Qualitative | requires_prediction_artifacts | all |  |  | confusion pairs from predicted and GT labels |
| E2 | confusion communities | Qualitative | requires_prediction_artifacts | all |  |  | class confusion graph |
| E2 | novel-vs-base class split | Qualitative | requires_taxonomy | all |  |  | per-class IoU plus base/novel mapping |
| E3 | cross-target rank stability | Group (ii); novelty-driving | computed_when_enough_methods_complete | all |  |  | method rankings across targets |
| E3 | source/target gap | Group (i) | requires_trained_reference_scores | all |  |  | trained-reference source and target mIoU |
| E3 | per-target Delta vs E1 | Group (ii) | requires_transfer_protocol | all |  |  | E1 target scores and E3 transfer target scores |
| E3 | per-target Delta MCMR | Group (ii) | requires_region_artifacts | all |  |  | MCMR@0.5 on E1/E2 and E3 targets |
| E3 | class-overlap-stratified mIoU | Group (ii); novelty-driving | requires_taxonomy | all |  |  | per-class IoU plus source-vocabulary overlap mapping |
| E3 | domain-shift sensitivity | Group (ii); novelty-driving | requires_embedding_artifacts | all |  |  | target mIoU drops plus CLIP/DINO dataset embedding distances |
| E3 | transferability of MCMR@0.5 | Group (ii) | requires_region_artifacts | all |  |  | MCMR@0.5 across E1/E2/E3 targets |
| E4 | training time | standard cost reporting | not_applicable_for_training_free_or_requires_refs | all |  |  | trained-reference training logs |
| E4 | offline preprocessing cost | standard cost reporting | requires_method_stage_logs | all |  |  | offline stage timing logs |
| E4 | trainable parameters | standard cost reporting | requires_model_introspection | all |  |  | official model parameter count |
| E4 | model calls | standard cost reporting | requires_method_stage_logs | all |  |  | instrumented method call counters |
| E4 | mIoU per GFLOP | Group (ii); novelty-supporting | requires_flops | all |  |  | mIoU plus FLOPs |
| E4 | offline amortization curve | Group (ii); novelty-supporting | requires_method_stage_logs | all |  |  | offline cost, online cost, evaluation set size N |
| E4 | per-stage memory/latency breakdown | Group (i) | requires_method_stage_logs | all |  |  | encoder/proposal/naming/post-processing stage timings |
| extra | prediction label entropy | extra discovery | requires_prediction_artifacts | all |  |  | prediction label maps |
| extra | rare-class rescue score | extra discovery | requires_taxonomy | all |  |  | per-class IoU plus train-frequency bins |
| extra | near-tie robustness | extra discovery | requires_score_artifacts | all |  |  | top-2 class scores or logits |

## Recent Running Or Failed Logs

- `failed` naclip/ade847/single 7.5%: `runs/logs/official_naclip_ade847_e2_oom_before_softmax_skip.log`
- `running_or_partial` proxyclip/context59/single 6.9%: `runs/logs/official_proxyclip_context59.log`
- `interrupted` proxyclip/voc20/single 55.2%: `runs/logs/official_proxyclip_voc20.log`
- `failed` sclip/ade847/single 95.0%: `runs/logs/official_sclip_ade847_e2_oom_before_softmax_skip.log`
