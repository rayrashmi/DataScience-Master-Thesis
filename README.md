🇬🇧 Social Media Toxicity Classification (Brexit Tweets)

This project builds a machine learning and deep learning pipeline to classify toxic vs non-toxic tweets in the context of the UK Brexit debate. It includes preprocessing, weak-supervision labeling, feature extraction, model training, and performance evaluation.

**** Key Features *****

TF–IDF based text representation
Logistic Regression baseline
Multi-Layer Perceptron (Deep Learning)
Confusion matrices, ROC curves, F1-scores
Lexicon-based weak supervision for labeling
Exploratory analysis of toxic vs non-toxic language

**** Results (Summary) *****

Model	                 Accuracy	      F1-Score	    ROC-AUC
Logistic Regression	     75%	           72%	        0.78
MLP (Deep Learning)    	 80%	           78%	        0.86

MLP outperforms the baseline, especially for context-dependent toxicity.

*** Tech Stack ****

Python · Scikit-learn · TensorFlow/Keras · NLTK · Pandas · NumPy · Matplotlib

**** Project Components ******

Preprocessing & cleaning
TF–IDF feature extraction
Training scripts for LR & MLP
Evaluation metrics & visualizations

**** Future Enhancements ******

Transformer models (BERT, RoBERTa)
Cross-platform toxicity analysis
Real-time toxicity monitoring dashboard
