# on2logic
Deep similarity searching of archive manuscripts

This repo contains code for calculating the visual similarity between images by using pre-trained computer vision ML models. We apply this to images downloaded from CUDL IIIF manifests and computationally infer connections between archive items. 

```mermaid
flowchart TD
Select[Select archives of interest] --> IIIF[Obtain IIIF manifests for archive items]
IIIF --> Image[Download images from manifest]
IIIF --> Metadata[Download metadata from manifest]
Image -- Pre-trained ML model --> Rep[Vector representation]
Rep --> Visualisation[Visualise results]
Visualisation --> Clustering[Cluster items by representation]
Visualisation --> Similarity[Inspect similarity between items]
Visualisation --> Contrast[Contrast computational connections with metadata-based connections]
Metadata ----> Contrast
```
