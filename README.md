# SpaDA

SpaDA (Spatially aware domain adaptation) is a python tool for the deconvolution of spatial transcriptomics.

## Abstract

Spatially Resolved Transcriptomics (SRT) offers unprecedented opportunities to elucidate the cellular arrangements within tissues. Nevertheless, the absence of deconvolution methods that simultaneouslymodel multi-modal features has impeded progress in understanding cellular heterogeneity in spatial contexts. To address this issue, we developed SpaDA, a novel spatially aware domain adaptation method that integrates multi-modal data (i.e., transcriptomics, histological images, and spatial locations) from SRT to accurately estimate the spatial distribution of cell types. SpaDA utilizes a self-expressive variational autoencoder, coupled with deep spatial distribution alignment, to learn and align spatial and graph representations from spatial multi-modal SRT data and single-cell RNA sequencing (scRNA-seq) data. This strategy facilitates the transfer of cell type annotation information across these two similarity graphs, thereby enhancing the prediction accuracy of cell type composition. Our results demonstrate that SpaDA surpasses existing methods in cell type deconvolution and the identification of cell types and spatial domains across diverse platforms. Moreover, SpaDA excels in identifying spatially colocalized cell types and key marker genes in regions of low-quality measurements, exemplified by high-resolution mouse cerebellum SRT data. In conclusion, SpaDA offers a powerful and flexible framework for the analysis of multi-modal SRT datasets, advancing our understanding of complex biological systems.

## Software dependencies

The dependencies for the codes are listed in requirements.txt

* anndata>=0.7.5
* leidenalg
* numpy>=1.17.0
* pandas>=1.0
* python==3.7
* scanpy>=1.6
* scikit-learn>=0.21.2
* scikit-misc>=0.1.3
* torch>=1.7.0
* tqdm>=4.56.0

## Tutorials

The tutorial for deconvolution of 10x Visium mouse cortex dataset:
https://github.com/zccqq/SpaDA/blob/main/tutorial_mouse_cortex.ipynb

The spatial transcriptomics dataset "Mouse Brain Serial Section 1 (Sagittal-Anterior)" is available on [10x Genomics website](https://www.10xgenomics.com/datasets/mouse-brain-serial-section-1-sagittal-anterior-1-standard-1-0-0).
The reference scRNA-seq dataset is available on the original publication [Adult mouse cortical cell taxonomy revealed by single cell transcriptomics](https://www.nature.com/articles/nn.4216).
