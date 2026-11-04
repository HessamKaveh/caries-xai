# Project definition

**Deep Learning for Dental Caries Detection Using Intraoral Images: Datasets, Models, and Explainability**

## 1. Motivation

Caries is the most common chronic disease worldwide. Intraoral photographs
taken with a camera or smartphone are cheap and need no radiation, so they
make teledentistry and screening possible. Many deep-learning papers report
high accuracy on such photos. Three problems recur in that literature:

1. **Data leakage.** One patient usually contributes several photos. Splitting
   by image puts the same mouth in both train and test, which inflates the
   reported metrics.
2. **Label granularity.** Fine-grained schemes such as Black's classes 1-6
   spread an already small number of lesions across rare classes. For
   screening, the clinically relevant question is whether caries is present
   and where.
3. **Explainability is shown, not measured.** Heat maps (usually Grad-CAM) are
   shown for a few hand-picked images. They are rarely checked against expert
   lesion masks or against a trivial "look at the centre" baseline, and their
   faithfulness to the model is rarely tested.

## 2. Research questions

| ID  | Question | Experiment |
|-----|----------|------------|
| RQ1 | How large is the gap between an image-random split and a patient-level split? | Same classifier trained and tested under both splits |
| RQ2 | Does merging Black's 6 caries classes into one "Caries" class improve caries detection? | YOLO11-seg trained on 9 vs 4 classes, both scored on the same unified caries task |
| RQ3 | How well do a lesion-level detector and a crop-level classifier screen for caries? | Detector (image-level) and ResNet-50 / EfficientNet-B0 (crop-level) ROC analysis |
| RQ4 | Which post-hoc explanation methods agree with expert lesion masks, and which are faithful to the model? | Grad-CAM, Grad-CAM++, Integrated Gradients and Occlusion, compared with Centre-prior and Random baselines |
| RQ5 | Are the classifier's probabilities calibrated, and can uncertain cases be referred to a dentist safely? | Temperature scaling, ECE, risk-coverage |
| RQ6 | Is performance consistent across sex and age groups? | Subgroup AUROC with patient-bootstrap 95 % CIs |

## 3. Data

**AlphaDent** (Sosnin et al., 2025; Apache-2.0): about 1,300 labelled
intraoral DSLR photographs from 295 patients, with polygon masks for 9
classes (abrasion, filling, crown, caries Black classes 1-6). The official
test split has no public labels, so this project merges the labelled
train and valid images and re-splits them **by patient**, stratified by
sex × age band (<30, 30-44, ≥45). The result is 70 % train, 10 % val and
20 % test, and no patient appears in more than one split.

## 4. Methods

* **Detectors:** YOLO11n-seg at 1024 px, 80 epochs with cosine LR and
  early stopping. One model is trained on 9 classes (`yolo9`) and one on the
  unified 4 classes (`yolo4`).
* **Label-space-independent evaluation:** the 9-class model's caries
  predictions are mapped to "Caries", so both detectors are scored by the same
  code (caries AP50, AP50:95, lesion recall by size tercile, and image-level
  screening AUROC).
* **Crop classifiers:** ImageNet-pretrained ResNet-50 and EfficientNet-B0
  trained on 224 px crops. Positive crops contain caries. Hard negatives are
  centred on fillings, crowns or abrasions with no caries pixels. Random
  negatives come from caries-free areas. Every crop keeps its caries mask.
* **Explainability audit** on correctly and incorrectly classified test
  caries crops, with these metrics:
  * pointing game: is the saliency peak within 8 px of the lesion?
  * relevance mass and lift: the share of saliency inside the lesion,
    divided by the lesion's share of the crop
  * pixel AUROC: the saliency map used as a lesion segmenter
  * deletion and insertion AUC: faithfulness under blurred-pixel removal
* **Calibration and selective prediction:** temperature scaling fitted on the
  validation split, ECE with 15 bins, risk-coverage curves and AURC.
* **Statistics:** 95 % CIs from a patient-level cluster bootstrap with 1000
  resamples.

## 5. What is new compared with existing work

* The AlphaDent paper reports 9-class YOLO baselines on its own split. This
  project adds a patient-stratified re-split, a common caries-unified scoring
  of the 9- and 4-class detectors, and a measured estimate of leakage.
* XAI in caries-photo papers is mostly qualitative. Here the explanations are
  scored against pixel-level expert masks and by perturbation faithfulness,
  with trivial baselines that each method has to beat.
* The project also covers calibration, referral (selective prediction) and
  demographic subgroup analysis on intraoral photos.

## 6. Limitations (stated up front)

* The data comes from a single centre with a single camera type. No external
  validation set exists.
* The crop-level task is easier than whole-mouth screening, because crops are
  sampled around annotations.
* Some subgroups (Caries class 4 and 6) are very small, and class 4 has no
  test instances after the patient split.
