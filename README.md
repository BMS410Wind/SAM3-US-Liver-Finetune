# SAM3-US-Liver-Finetune

A fine-tuning implementation of Meta's Segment Anything Model 3 (SAM3) for ultrasound liver image segmentation.

## 📋 Overview

This project adapts the SAM3 model for medical imaging applications, specifically targeting liver segmentation in ultrasound images. The implementation includes custom model builders and utilities for working with SAM3 in a medical imaging context.

## 🚀 Features

- **SAM3 Integration**: Built on Meta's SAM3 architecture
- **Medical Imaging Optimized**: Specialized for ultrasound liver imaging

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/BMS410Wind/SAM3-US-Liver-Finetune.git
cd SAM3-US-Liver-Finetune

# Install SAM3 package
pip install -e .
```

## 🏗️ Project Structure

```
SAM3-US-Liver-Finetune/
├── sam3.egg-info/          # Package metadata
├── sam3/                   # Main package directory
├── finetuneV4.py          # Fine-tuning script
├── infer_sam_liverfinetuneV4.py  # Web-based inference
├── labeltool.py           # Labeling tool
├── liverpred.png          # Sample prediction
└── LICENSE                # License file
```

## 🎯 Usage

### Fine-tuning

```bash
python finetuneV4.py 
```

### Inference

```bash
infer_sam_liverfinetuneV4.py 
```

## 🙏 Acknowledgments

- Meta AI Research for the SAM3 model
- The original SAM3 repository: [facebookresearch/sam3](https://github.com/facebookresearch/sam3)

## 📧 Contact

- For questions or issues, please open an issue on the [GitHub repository](https://github.com/BMS410Wind/SAM3-US-Liver-Finetune/issues).
---
