# Third-party licences and attribution

ForensicVision itself is Apache-2.0 (see `LICENSE`).

Every neural architecture in this repository is an **independent
implementation written from the published papers and the public reference
code's layer naming**, so that official upstream checkpoints load unmodified.
No upstream source file has been copied. The upstream projects are nevertheless
credited below, and their licences govern the **weight files**, which the
application downloads from the upstream release assets on your explicit request
and never redistributes.

> **Before deploying:** several weight sets are restricted to non-commercial
> research. Confirm the terms of every model you enable for your intended use.
> Where a licence is ambiguous, the corresponding entry says so.

---

## Runtime dependencies

| Package | Licence | Use |
|---|---|---|
| [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) | GPL v3 / commercial | Entire GUI |
| [Qt 5](https://www.qt.io/) | LGPL v3 | Underlying toolkit |
| [NumPy](https://numpy.org/) | BSD-3-Clause | Array operations |
| [OpenCV](https://opencv.org/) (`opencv-python`) | Apache-2.0 | Image I/O, filtering, CLAHE, NLM, bilateral, guided filter |
| [Pillow](https://python-pillow.org/) | MIT-CMU | Fallback decoding, quantisation tables |
| [scikit-image](https://scikit-image.org/) | BSD-3-Clause | Supporting image utilities |
| [SciPy](https://scipy.org/) | BSD-3-Clause | Numerical support |
| [PyTorch](https://pytorch.org/) | BSD-3-Clause | Neural inference |
| [torchvision](https://pytorch.org/vision/) | BSD-3-Clause | Tensor utilities |
| [SQLAlchemy](https://www.sqlalchemy.org/) | MIT | Case database |
| [ReportLab](https://www.reportlab.com/) | BSD-3-Clause | PDF reports |
| [ExifRead](https://github.com/ianare/exif-py) | BSD-3-Clause | EXIF extraction |
| [psutil](https://github.com/giampaolo/psutil) | BSD-3-Clause | System information |

### PyQt5 licensing

**PyQt5 is GPL v3.** Distributing a binary that links it obliges you to offer
the complete corresponding source of the whole work under the GPL, or to hold a
commercial PyQt licence from Riverbank Computing. This affects redistribution,
not internal use.

## Optional dependencies

| Package | Licence | Use |
|---|---|---|
| [pytesseract](https://github.com/madmaze/pytesseract) | Apache-2.0 | OCR binding |
| [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) | Apache-2.0 | OCR engine (separate native install) |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Apache-2.0 | Alternative OCR engine |
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | **AGPL-3.0** / commercial | Object detection |
| [pytest](https://pytest.org/) | MIT | Testing |
| [pypdf](https://github.com/py-pdf/pypdf) | BSD-3-Clause | Report verification in tests |
| [PyInstaller](https://pyinstaller.org/) | GPL-2.0 with bundling exception | Packaging |

### Ultralytics licensing

**Ultralytics YOLO is AGPL-3.0.** The AGPL's network clause is triggered by
offering the software as a network service. Object detection is optional and is
not installed by default; if you enable it, review the AGPL or obtain an
Ultralytics commercial licence.

---

## Model architectures and weights

### Real-ESRGAN

- **Authors:** Xintao Wang, Liangbin Xie, Chao Dong, Ying Shan (Tencent ARC)
- **Repository:** https://github.com/xinntao/Real-ESRGAN
- **Paper:** *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure
  Synthetic Data*, ICCVW 2021
- **Code licence:** BSD-3-Clause
- **Weight licence:** BSD-3-Clause
- **Weights:** `RealESRGAN_x4plus.pth` (v0.1.0), `RealESRGAN_x2plus.pth`
  (v0.2.1), `RealESRGAN_x4plus_anime_6B.pth` (v0.2.2.4), from the repository's
  GitHub release assets
- **Implementation:** `restoration/realesrgan/arch.py` - RRDBNet written from
  the paper with upstream layer naming (`conv_first`, `body.N.rdbM.convK`,
  `conv_body`, `conv_up1/2`, `conv_hr`, `conv_last`)
- **Note:** upstream publishes no SHA-256 for these assets, so downloads cannot
  be verified against a published digest. The digest of what is received is
  recorded.

### SwinIR

- **Authors:** Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang,
  Luc Van Gool, Radu Timofte (ETH Zurich)
- **Repository:** https://github.com/JingyunLiang/SwinIR
- **Paper:** *SwinIR: Image Restoration Using Swin Transformer*, ICCVW 2021
- **Code licence:** Apache-2.0
- **Weight licence:** Apache-2.0
- **Weights:** `001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth`,
  `003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth`,
  `005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth`,
  `006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth` (release v0.0)
- **Implementation:** `restoration/swinir/arch.py` - written from the paper;
  the `timm` helpers (`DropPath`, `trunc_normal_`, `to_2tuple`) are reproduced
  so `timm` is not a dependency

### Restormer

- **Authors:** Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat,
  Fahad Shahbaz Khan, Ming-Hsuan Yang, Ling Shao
- **Repository:** https://github.com/swz30/Restormer
- **Paper:** *Restormer: Efficient Transformer for High-Resolution Image
  Restoration*, CVPR 2022
- **Code licence:** ACADEMIC / non-commercial research use
- **Weight licence:** **Non-commercial research use only**
- **Weights:** `motion_deblurring.pth`,
  `single_image_defocus_deblurring.pth`, `real_denoising.pth` (release v1.0)
- **Implementation:** `restoration/restormer/arch.py` - MDTA and GDFN written
  from the paper; `einops.rearrange` replaced with equivalent
  `reshape`/`permute`, so `einops` is not a dependency
- **Note:** review the upstream terms before any commercial or operational
  deployment.

### FBCNN

- **Authors:** Jiaxi Jiang, Kai Zhang, Radu Timofte (ETH Zurich)
- **Repository:** https://github.com/jiaxi-jiang/FBCNN
- **Paper:** *Towards Flexible Blind JPEG Artifacts Removal*, ICCV 2021
- **Code licence:** Apache-2.0
- **Weight licence:** Apache-2.0
- **Weights:** `fbcnn_color.pth`, `fbcnn_gray.pth` (release v1.0)
- **Implementation:** `restoration/fbcnn/arch.py` - written from the paper,
  reproducing the KAIR `basicblock` naming convention the checkpoints use

### DnCNN

- **Authors:** Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, Lei Zhang
- **Repository:** https://github.com/cszn/KAIR
- **Paper:** *Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for
  Image Denoising*, IEEE TIP 2017
- **Code licence:** MIT
- **Weight licence:** MIT
- **Weights:** `dncnn_color_blind.pth`, `dncnn_25.pth` (KAIR release v1.0)
- **Implementation:** `restoration/dncnn/arch.py`

### NAFNet

- **Authors:** Liangyu Chen, Xiaojie Chu, Xiangyu Zhang, Jian Sun (MEGVII)
- **Repository:** https://github.com/megvii-research/NAFNet
- **Paper:** *Simple Baselines for Image Restoration*, ECCV 2022
- **Code licence:** MIT
- **Weight licence:** released for non-commercial research; review upstream
- **Weights:** distributed via Google Drive only. ForensicVision declares no
  download URL for them and requires manual installation, because a weight file
  whose provenance cannot be stated is not admissible tooling.
- **Implementation:** `restoration/nafnet/arch.py`

### CodeFormer

- **Authors:** Shangchen Zhou, Kelvin C.K. Chan, Chongyi Li,
  Chen Change Loy (S-Lab, NTU)
- **Repository:** https://github.com/sczhou/CodeFormer
- **Paper:** *Towards Robust Blind Face Restoration with Codebook Lookup
  Transformer*, NeurIPS 2022
- **Code licence:** **S-Lab License 1.0 - NON-COMMERCIAL research use only**
- **Weight licence:** **S-Lab License 1.0 - NON-COMMERCIAL research use only**
- **Weights:** `codeformer.pth` (release v0.1.0)
- **Implementation:** `restoration/codeformer/arch.py` - the VQGAN
  autoencoder, codebook-lookup transformer and controllable feature
  transformation blocks written from the paper with upstream layer naming
- **Note:** the non-commercial restriction is significant. Confirm that your
  intended use is permitted before enabling this model.

### YuNet face detector

- **Author:** Wei Wu, Shenzhen Institute of Advanced Technology; distributed
  through the OpenCV Zoo
- **Repository:** https://github.com/opencv/opencv_zoo
- **Licence:** MIT
- **Weights:** `face_detection_yunet_2023mar.onnx` (227 KiB)
- **Use:** locates faces and emits the five landmarks CodeFormer's alignment
  stage needs. The inference API (``cv2.FaceDetectorYN``) is part of OpenCV
  itself, so no additional Python dependency is introduced.
- **Note:** used only for detection and alignment. No face recognition,
  matching or identification is performed anywhere in this application.

### GFPGAN *(declared, not integrated)*

- **Authors:** Xintao Wang, Yu Li, Honglun Zhang, Ying Shan (Tencent ARC)
- **Repository:** https://github.com/TencentARC/GFPGAN
- **Paper:** *Towards Real-World Blind Face Restoration with Generative Facial
  Prior*, CVPR 2021
- **Code licence:** Apache-2.0
- **Status:** not executable in this build; see `docs/LIMITATIONS.md`

### LaMa *(declared, not integrated)*

- **Authors:** Roman Suvorov et al. (Samsung AI Center Moscow)
- **Repository:** https://github.com/advimman/lama
- **Paper:** *Resolution-robust Large Mask Inpainting with Fourier
  Convolutions*, WACV 2022
- **Code licence:** Apache-2.0; **weights CC BY-NC-SA 4.0**
- **Status:** not executable in this build; see `docs/LIMITATIONS.md`

---

## Algorithms implemented from the literature

Implemented directly in this repository from the published descriptions. No
third-party code was copied; the citations are academic attribution.

| Algorithm | Reference | Used in |
|---|---|---|
| Perceptual blur metric | Crete et al., SPIE HVEI 2007 | `analysis/blur.py` |
| Laplacian focus measure | Pech-Pacheco et al., ICPR 2000 | `analysis/blur.py` |
| Fast noise variance estimation | Immerkaer, CVIU 64(2), 1996 | `analysis/noise.py` |
| Wavelet MAD sigma estimator | Donoho & Johnstone, Biometrika 1994 | `analysis/noise.py` |
| Blocking artefact measurement | Wang, Bovik & Evan, ICIP 2000 | `analysis/jpeg.py` |
| IJG quantisation scaling | Independent JPEG Group reference implementation | `analysis/jpeg.py` |
| Dark channel prior | He, Sun & Tang, CVPR 2009 | `analysis/haze.py`, `restoration/classical/enhance.py` |
| Guided filter | He, Sun & Tang, ECCV 2010 | `restoration/classical/enhance.py` |
| Richardson-Lucy deconvolution | Richardson, JOSA 1972; Lucy, AJ 1974 | `restoration/classical/deconvolution.py` |
| Wiener filtering | Wiener, 1949 | `restoration/classical/deconvolution.py` |
| CLAHE | Zuiderveld, *Graphics Gems IV*, 1994 | via OpenCV |
| Non-local means | Buades, Coll & Morel, CVPR 2005 | via OpenCV |
| Bilateral filter | Tomasi & Manduchi, ICCV 1998 | via OpenCV |
| Lanczos resampling | Duchon, *J. Appl. Meteorology*, 1979 | via OpenCV |

---

## Cryptographic primitives

SHA-256, SHA-512 and MD5 come from Python's standard `hashlib`. MD5 is computed
for cross-reference with legacy case-management systems only; it is
collision-broken and is never used by this application for integrity
verification. Every API that exposes it says so.

---

## Reporting a licensing problem

If any attribution here is incomplete or incorrect, please open an issue. The
intent is full compliance and full credit.
