# Matrix Factorization Applications

This repository contains the implementation and analysis focusing on matrix factorization techniques. It explores Non-Negative Matrix Factorization (NMF) in two contexts: text classification for BBC news articles (Part 1) and recommender systems using movie ratings (Part 2). Both parts demonstrate unsupervised learning approaches, with comparisons to supervised methods and discussions on limitations and improvements.

The notebooks are self-contained Jupyter files with code, explanations, visualizations, and results. They were developed in Python 3.11 using libraries like scikit-learn, pandas, and NLTK.

## Overview

Matrix factorization is a powerful technique for dimensionality reduction and pattern discovery in data. In this lab:

-   **Part 1** applies NMF to classify BBC news articles into categories (e.g., business, tech) by treating it as a topic modeling task. It includes exploratory data analysis (EDA), model tuning, ensemble methods, and a comparison with supervised logistic regression.
-   **Part 2** investigates the limitations of scikit-learn's NMF implementation for recommender systems on movie rating data. It covers data preparation, model fitting, RMSE evaluation, hyperparameter tuning, and suggestions for addressing shortcomings like handling sparsity.

These notebooks align with practical goals to understand NMF's strengths in unsupervised scenarios, its performance relative to supervised alternatives, and practical fixes for real-world issues.

Key datasets:

-   BBC News (from Kaggle: \~1490 train articles, \~735 test articles across 5 categories).
-   Movie ratings (from a provided CSV: \~700k train entries, \~300k test entries).

Results highlight NMF's effectiveness for topic extraction (\~95% accuracy in classification) but reveal challenges in recommender systems (high RMSE \~2.7 due to sparsity handling).

## Repository Structure

-   `Week-4-Lab1-Part-1-v3.2.ipynb`: Part 1 - BBC News Classification.
-   `Week-4-Lab-Part2-v1.1.ipynb`: Part 2 - NMF Limitations in Recommender Systems.
-   `README.md`: This file.
-   Data files: Place `BBC_News_Train.csv`, `BBC_News_Test.csv`, `train.csv`, and `test.csv` in `data/`.

## Requirements

To run these notebooks, you'll need:

-   Python 3.8+ (tested on 3.11).
-   Jupyter Notebook or JupyterLab.
-   Key libraries:
    -   pandas
    -   numpy
    -   scikit-learn
    -   matplotlib
    -   seaborn
    -   nltk (with 'stopwords' and 'punkt' corpora downloaded)
    -   scipy

Install dependencies via pip:

```
pip install pandas numpy scikit-learn matplotlib seaborn nltk scipy
```

For NLTK resources, run in a Python shell or notebook cell:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## Setup Instructions

1.  Clone this repository:

```
git clone https://github.com/your-username/week-4-lab-matrix-factorization.git
cd week-4-lab-matrix-factorization
```

2.  Download datasets if needed:
    -   BBC News: From [Kaggle](https://www.kaggle.com/c/learn-ai-bbc/overview).
    -   Movie ratings: Assumed from course resources (e.g., HW3 recommender setup); place in the notebook's referenced paths (e.g., `C:/CUBoulder/MSAI/CSCA5632/Week-3/` – update to your local paths).
3.  Launch Jupyter:

```
jupyter notebook
```

Open and run the notebooks sequentially.

Note: Update file paths in the notebooks to match your local setup (e.g., CSV loading cells use absolute paths – change to relative if preferred).

## Usage

### Running Part 1 (BBC News Classification)

-   Open `Week-4-Lab1-Part-1-v3.2.ipynb`.
-   Execute cells top-to-bottom.
-   Key sections:
    -   EDA: Data inspection, cleaning, visualizations (e.g., category distribution, word frequencies).
    -   Modeling: TF-IDF vectorization + NMF for topic modeling; label mapping via mode; hyperparameter tuning (n_components=5-10).
    -   Improvements: CountVectorizer variant, data subset efficiency, ensemble with alignment.
    -   Outputs: Accuracy (\~0.95), confusion matrix, submission CSV for Kaggle.
-   Expected runtime: \~5-10 minutes on a standard laptop.

### Running Part 2 (NMF Limitations in Recommenders)

-   Open `Week-4-Lab-Part2-v1.1.ipynb`.
-   Execute cells top-to-bottom.
-   Key sections:
    -   Data loading and pivoting to user-item matrices.
    -   NMF application with KL loss; RMSE calculation on test data.
    -   Tuning: n_components (10-30); imputation with user means.
    -   Discussion: High RMSE (\~2.7) due to zero-handling; suggestions like bias addition or hybrids.
-   Outputs: RMSE table, best parameters (e.g., n=30, RMSE\~2.71).
-   Expected runtime: \~10-15 minutes (NMF fitting can be iterative).

For both:

-   Visualizations (e.g., histograms, confusion matrices) render inline.
-   Tune hyperparameters in marked cells for experimentation.
-   Generate Kaggle submissions directly from notebooks.

## Results and Insights

-   **Part 1**: Unsupervised NMF achieves \~95% accuracy, close to supervised (\~97%), showing strong topic separation. Ensemble and subsets maintain performance, emphasizing data efficiency.
-   **Part 2**: NMF underperforms (RMSE \~2.7 vs. baselines \~0.9-1.0) due to sparsity and lack of bias terms. Fixes like imputation help marginally.

Detailed discussions in notebooks cover why results occur (e.g., TF-IDF vs. CountVec) and propose enhancements (e.g., lemmatization, GPU acceleration).

## Contributing

If you'd like to contribute (e.g., add alternative models like SVD), fork the repo and submit a pull request. Ensure code is commented and follows PEP8.

## License

This project is licensed under the MIT License - see the <LICENSE> file for details (or add one if needed).

## Acknowledgments

-   Based on CSCA-5632 course materials.
-   Datasets from Kaggle and course-provided sources.
-   Libraries: Thanks to scikit-learn and NLTK communities.

If you encounter issues, check Jupyter console logs or open an issue on GitHub. This README is designed for clarity and reproducibility in educational settings.
