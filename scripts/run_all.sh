#!/usr/bin/env bash
# Full CariesXAI pipeline, end to end. RESUMABLE: every stage skips itself if
# its outputs exist, YOLO continues from runs/seg/*/weights/last.pt and the
# classifiers from weights/cls_*_last.pt. Safe to kill and re-run at any time.
# Expects AlphaDent unpacked at $CARIES_RAW (default ~/caries-xai-data/raw/alphadent).
set -euo pipefail
cd "$(dirname "$0")/../src"
PY=${PY:-python}
EPOCHS=${EPOCHS:-80}
stage() { echo "STAGE $1 $(date +%H:%M)"; }

[ -f ../data/meta.csv ] || $PY prepare_data.py --seed 42;            stage data_done
[ -f ../data/crops/crops.csv ] || $PY make_crops.py;                stage crops_done
$PY train_yolo.py --labels yolo9 --epochs "$EPOCHS";                stage yolo9_done
for arch in resnet50 efficientnet_b0; do
  for sp in patient random; do   # "random" = leakage experiment only
    $PY train_cls.py --arch $arch --split $sp;                      stage cls_${arch}_${sp}_done
  done
done
[ -f ../results/xai_summary.csv ] || $PY xai_eval.py --n 300;       stage xai_done
$PY train_yolo.py --labels yolo4 --epochs "$EPOCHS";                stage yolo4_done
$PY eval_detection.py --weights ../weights/yolo9_yolo11n-seg_1024.pt ../weights/yolo4_yolo11n-seg_1024.pt; stage eval_det_done
$PY visualize_detection.py;                                         stage viz_done
$PY analysis.py;                                                    stage ALL_DONE
