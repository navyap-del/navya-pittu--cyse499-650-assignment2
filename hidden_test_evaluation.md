## Stage 2: Hidden Test Evaluation

Model used: the exact Stage 1 checkpoint (model_checkpoint/vectorizer.joblib and model_checkpoint/model.joblib LogisticRegression classifier over TF-IDF and VADER features).
No retraining, fine-tuning, or modification was performed, this notebook is inference only, as required.


## Results

Public test accuracy was 0.6250 with a macro F1 of 0.6110, on 400 examples. Hidden test accuracy was 0.6133 with a macro F1 of 0.5943, on 600 examples.

Hidden test confusion matrix (rows are the true label, columns are predicted, order is negative then positive): 

[[119 181]
[ 51 249]]

## Discussion

Hidden test performance is close to, and only slightly below, public test performance, about a 1.2 point drop in accuracy and a 0.017 drop in macro F1. Both splits show the same pattern, strong recall on positive reviews around 0.81 to 0.83, but noticeably weaker recall on negative reviews around 0.40 to 0.43, so the model consistently over predicts positive. This matches what
I expected from Stage 1, since the training set only had 60 negative reviews out of 240 total, which isn't enough examples to learn the full range of ways critics express negative sentiment.

The small, consistent gap between public and hidden test, rather than a sharp drop, is a good sign. It means the model's generalization is stable on genuinely unseen data, not just tuned to the specific quirks of public_test.csv. The negative recall weakness showing up in both unseen sets confirms this is a data problem, too few negative training examples, rather than an overfitting to one test set problem.

## What I'd try next with more time/compute

- Fine-tune a pretrained transformer (e.g. DistilBERT). This wasn't possible in the Stage 1 development environment (no access to download pretrained weights), but its subword tokenizer and pretrained language understanding would likely help substantially, especially given how much of the test vocabulary never appears in training.
- Get more negative training examples - every diagnostic run (CV vs. public test gap, public vs. hidden test gap, persistently low negative recall) points to training data imbalance/size as the dominant bottleneck, not model architecture.
- Use data augmentation (e.g. back-translation) on the 60 negative reviews instead of plain duplication, so the model sees varied phrasing rather than memorizing the same reviews.
- Combine multiple sentiment lexicons as additional engineered features to further reduce reliance on the small training vocabulary.