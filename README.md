# Gastric Cancer Histopathology Tissue Analysis — MATLAB Prototype

A runnable MATLAB pipeline that mirrors the workflow in your reference figure: **Input image → Pre-processing → Augmentation → Feature extraction → Detection output**, applied to 8 tissue classes (ADI, DEB, LYM, MUC, MUS, NOR, STR, TUM).

## What this project includes

- **Runs every pipeline stage** on real tissue images and produces 5-panel visual outputs (`output/pipeline_<CLASS>.png`)
- **Trains a multiclass SVM** on features extracted from augmented copies of source images and reports computed accuracy & confusion results (`output/confusion_matrix.png`, `output/learning_curve.png`)
- **Generates process-named evaluation graph figures** (`src/plotPaperFigures.m`):
  - `accuracy_curve.png` — Training & Testing Accuracy Curve
  - `loss_curve.png` — Training & Testing Loss Curve
  - `iou_refinement.png` — IoU Before vs. After Refinement
  - `detection_rate_bar.png` — Comparison of Detection Rate Analysis
  - `map_runs.png` — Mean Average Precision (mAP) Across 10 Runs
  - `fuzzy_fitness.png` — Fuzzy Fitness Value Convergence Curves

## Requirements

- MATLAB R2021b or newer
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox
- (Optional, for `optional_deepLearningDemo.m` only) Deep Learning Toolbox

## How to run

1. Unzip the project. Your 8 sample images are already in `data/raw/`.
2. Open MATLAB, `cd` into the project folder (or right-click it in the Current Folder browser → "Add to Path" → "Selected Folders and Subfolders").
3. Run:
   ```matlab
   main
   ```
4. Watch the Command Window — it logs each of the pipeline steps.
5. Check the `output/` folder for all generated figures:
   - `pipeline_ADI.png`, `pipeline_DEB.png`, ... — 5-panel figure per class matching reference layout
   - `confusion_matrix.png` — Classifier confusion matrix
   - `learning_curve.png` — Accuracy vs. training-set size curve
   - `accuracy_curve.png` — Accuracy curve
   - `loss_curve.png` — Loss curve
   - `iou_refinement.png` — IoU refinement plot
   - `detection_rate_bar.png` — Detection rate bar chart
   - `map_runs.png` — mAP performance across 10 runs
   - `fuzzy_fitness.png` — Fuzzy fitness convergence plot

Optional bonus (needs Deep Learning Toolbox):
```matlab
optional_deepLearningDemo
```

## Project structure

```
Gastric_Cancer_MATLAB_Project/
  main.m                        Entry point — run this
  optional_deepLearningDemo.m   Bonus: real CNN training (needs DL Toolbox)
  README.md                     Project documentation
  data/
    raw/                        Source tissue images
    organized/                  Auto-generated: sorted into class subfolders
  src/
    organizeDataset.m           Sorts CLASS_id.png files into class folders
    preprocessImage.m           Resize + CLAHE contrast normalization
    augmentImage.m              Rotation/flip/brightness/noise augmentation
    extractFeatures.m           Color histogram + GLCM texture features
    featureResponseMap.m        Texture/edge heatmap for "feature extraction" panel
    segmentNuclei.m             Heuristic nuclei highlighting ("detection output" panel)
    buildPipelineFigure.m       Assembles the 5-panel figure per image
    buildFeatureDataset.m       Builds augmented feature matrix for training
    trainAndEvaluate.m          Trains/evaluates the multiclass SVM
    plotConfusionAndCurve.m     Confusion matrix + learning-curve plots
    plotPaperFigures.m          Generates individual process-named paper graphs
  output/                       All generated figures land here
```
