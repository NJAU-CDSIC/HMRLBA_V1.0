# HXK4 Prediction

This task preprocesses the HXK4 virtual screening dataset and predicts it using
a MEGDTA model trained on PDBbind.

Run from the MEGDTA SOTA package root:

```bash
PYTHONPATH=core/megdata:core \
python tasks/hxk4_prediction/preprocess_hxk4.py \
  --data_dir data/hxk4_raw \
  --output_dir data/hxk4
```

Then run prediction:

```bash
PYTHONPATH=core/megdata:core \
python tasks/hxk4_prediction/predict_hxk4.py \
  --dataset hxk4 \
  --model models_identity30/best_model_fold0.pth \
  --fold 0 \
  --gpu 0
```

Generated HXK4 predictions and metrics are local artifacts and should not be
committed.
