# EgoUse Medium-33

EgoUse predicts per-hand interaction timelines for 33 canonical objects from egocentric Aria RGB. The selected development pipeline uses a closed-set YOLO26m, 100DOH hand detections, 18-channel pair features and a causal pairwise GRU. Hysteresis turns each pair ON at probability ≥0.50 and OFF below 0.45. Outputs include logits, probabilities, per-hand predictions, merged OR and timestamped half-open intervals.

## Local setup

Use the exact Python environments and checkpoint hashes in [the release manifest](egouse/config/medium33-release.json). Create a local path file using [the template](egouse/config/medium33-assets.example.json). You need the pinned 100DOH source and compiled operators, its ResNet-101 weight, the fine-tuned Medium-33 detector and the selected Student checkpoint. The local release was validated with these assets on one desktop host. A fresh setup has not been tested.

The necessary Python packages are obtainable from their distributors. Use separate CUDA-enabled hand and object environments, and the pinned main Python for the CLI. Build the pinned 100DOH checkout's ROI/NMS operators in its compatible CUDA/PyTorch toolchain using the vendor build instructions. Obtain the exact authorized weights from their custodians and verify SHA-256. The two fine-tuned EgoUse weights are external project assets. Acquiring the code does not provide access to the trained weights, recordings or annotations. See [asset and license notes](egouse/docs/publication/ASSETS.public.md).

```bash
"$MAIN_PYTHON" egouse/release/cli.py --assets /path/to/assets.local.json \
  features /path/to/features.csv.gz --output /path/to/new-cached-run

CUDA_VISIBLE_DEVICES=0 "$MAIN_PYTHON" egouse/release/cli.py --assets /path/to/assets.local.json \
  vrs /path/to/recording.vrs --start 194 --count 90 --output /path/to/new-vrs-run
```

VRS mode supports 1408×1408 Aria RGB stream 214-1 with timecode timestamps. Cached mode requires the exact full deployment grid and blank interaction labels. Both modes are offline batch processing. The project does not establish arbitrary video ingestion, incremental features, or wearable runtime.

The development comparison used selection-exposed Teacher clips and three human-reviewed recordings with prior detector and project exposure. No matched human-reference aria_small event was obtained under the frozen event criterion. The top-level README and its original demo describe the historical system. This update distributes project source and setup instructions. Weights, recordings and private evaluation artifacts remain external.
