# Asset acquisition and terms

The release manifest pins three validated environments. Obtain the exact Python and PyTorch builds, NumPy, OpenCV and the listed optional vendor packages from their distributors. The main environment includes Project Aria Tools and pandas. The hand environment includes torchvision and the 100DOH checkout's locally compiled ROI/NMS extension. The object environment includes its own CUDA PyTorch and Ultralytics. The vendor hand build entry is lib/setup.py. Reproduce a compatible build rather than installing the vendor's historical CUDA versions into another runtime.

The upstream 100DOH implementation is ddshan/hand_object_detector, pinned by the manifest. That source ships under MIT. The specific 100DOH weight needs its own acquisition and redistribution review. The validated Ultralytics package metadata declares AGPL-3.0 with a v3-or-later classifier. Project Aria Tools declares Apache-2.0. External and locally fine-tuned weights have separate provenance. Their redistribution rights are not established by this code snapshot.

HOT3D recordings and annotations use separate dataset terms. In the retained dataset agreement sequence data and non-hand annotations have CC BY-SA terms, hand annotations have CC BY-NC-SA terms, and the 3D models have additional sale restrictions. Verify the terms attached to each asset before distributing it or a derivative. Private evaluation files and experiment media are excluded.

This scope is a source-only proposal. Repository-level licensing and any required upstream notices need owner approval before publication. The snapshot includes no vendor checkout, weights, data or recordings.
