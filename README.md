# MoA Prediction Project

This repository is based on work from the Kaggle **Mechanisms of Action (MoA) Prediction** competition: [lish-moa](https://www.kaggle.com/competitions/lish-moa).

`Python` · `Jupyter` · `pandas` · `NumPy` · `scikit-learn` · `PyTorch` · `TabNet` · `PCA` · `KMeans` · `Multi-Label Classification` · `Feature Engineering` · `Ensemble Modeling`

Mechanism of action prediction is one of the more interesting machine learning problems I have worked on in applied biology. Instead of trying to identify a single label from a neat, structured input, I start with a very large collection of gene-expression and cell-response measurements and ask a harder question: given how a compound changes a biological system, what is it probably doing?

That is what this project is about.

This repository contains my notebook-based pipeline for predicting a compound's likely **mechanism of action (MoA)** from high-dimensional biological response data. I built it as part of the Kaggle `lish-moa` setting, and the workflow combines feature engineering, dimensionality reduction, neural networks, TabNet experiments, and ensemble modeling to improve multi-label prediction performance.

For me, this project was also a learning process. It gave me a way to apply statistics, machine learning, feature engineering, and model evaluation to a real biological prediction problem instead of only studying those ideas in isolation. I got to work through things like noisy data, correlated features, sparse labels, validation strategy, and ensemble behavior in a much more hands-on way.

Ideally, this kind of project would be as simple as "feed in the biology data, train a model, get a clean answer." Of course, in practice, things are more complicated. Biological data is noisy, many features are strongly correlated, useful signals are often sparse, and a single compound can activate multiple mechanisms at once. So the real work is not just training a model. The real work is building a pipeline that makes the signal easier to learn.

That is the purpose of this repository.

## What This Project Does

At a high level, I built the system to:

- reads gene and cell response features for each compound treatment
- cleans and normalizes those signals
- creates additional engineered features that summarize broader biological patterns
- trains multi-label models to estimate likely mechanisms
- combines multiple model variants to produce more stable predictions

In other words, I take a raw biological fingerprint and turn it into a ranked set of mechanism predictions.

## Goal of the Project

The goal of building this system was to create a practical modeling pipeline that could take complex biological response data and convert it into useful mechanism-of-action predictions. More specifically, I wanted to answer a common applied ML question in biology:

given a compound's observed effect on genes and cells, can I build a model that helps infer what biological mechanisms are most likely driving that effect?

From a systems perspective, my goal was not just to train a single model, but to build a workflow that:

- improves signal quality before training
- captures broad statistical structure in the data
- supports stable multi-label prediction
- combines multiple modeling approaches rather than depending on one run

## Why This Problem Is Hard

Before getting into the workflow, there is one important point I want to make clear.

This is not a "regular" classification problem where each sample belongs to exactly one category. In the MoA setting, one treatment may be associated with several biological mechanisms at once. That means the model needs to produce **many probabilities per sample**, not just a single class label.

On top of that:

- the number of input features is large
- many features carry overlapping information
- some variables are noisy or weakly informative
- the biological effect is often distributed across patterns rather than isolated columns

Because of this, the pipeline is designed to strengthen signal before the final prediction stage.

## How the Pipeline Works

Now that I have that out of the way, I can walk through the core stages of the modeling workflow: input data, normalization, feature compression, clustering, neural network training, and ensembling.

### Input Data

Each sample contains a combination of:

- gene expression features
- cell response features
- treatment metadata such as dose and exposure time

Taken together, these form a high-dimensional description of how a biological system responded to a compound.

If I were only trying to train a quick baseline model, the raw columns would already be enough to get started. But if I want stronger performance, I generally need to do more than just feed them directly into a classifier.

### Signal Normalization

The first major step is normalization.

Raw biological measurements are often skewed and may contain extreme values that make optimization harder. To reduce this problem, the workflow uses **quantile-based transformations** so that features behave more consistently across the dataset.

This helps in a few ways:

- it reduces the influence of outliers
- it makes distributions easier for the model to work with
- it improves training stability

In short, before the model tries to learn biology, I first reshape the data into a form that is easier to learn from.

### Feature Compression with PCA

The next challenge is correlation. Gene and cell features often move together, which means the raw input space is much larger than the truly useful signal space.

To deal with this, I add **principal component analysis (PCA)** features for both gene and cell subsets. These PCA features act as compressed summaries of the strongest patterns in the data.

This does not replace the original inputs entirely. Instead, it gives the model another view of the data:

- the original detailed measurements
- a compressed summary of broad biological variation

That combination is often more useful than either one alone.

### Statistical Feature Engineering

In addition to PCA, I create aggregate statistical features across gene, cell, and combined feature sets.

These include measures such as:

- sums
- means
- standard deviations
- skewness
- kurtosis

The value of these features is that they capture the overall "shape" of a biological response. Two compounds may not match feature-by-feature, but they may still produce similar higher-level response profiles.

This part of the workflow leans on a few core statistical ideas:

- **distribution shaping** through quantile normalization
- **variance and covariance structure** through PCA
- **central tendency and dispersion** through means and standard deviations
- **distribution shape descriptors** through skewness and kurtosis
- **group similarity** through clustering

So while the final prediction layer is machine learning, a good portion of the pipeline is really about using statistical structure to make the learning problem cleaner and more informative. That was a big part of the learning value of this project for me too: I was not just training models, I was learning how statistical thinking shapes the quality of the ML pipeline.

### Cluster Features

Another useful trick in the workflow is clustering.

Using KMeans, I group samples with similar gene or cell response patterns and then feed those cluster identities back into the model as additional features.

This gives the model contextual information about which response family a sample resembles. That can be especially useful when biological effects are easier to recognize at a group level than at the level of individual measurements.

### Removing Weak Features

Not every column is worth keeping.

Some features vary so little that they add almost no predictive value. To reduce clutter, I use variance-based filtering to remove low-information inputs before model training.

This keeps the feature space more focused and can help reduce overfitting.

## Training the Prediction Models

Once the features are prepared, I train the main predictive models.

The core workflow uses **PyTorch neural networks** for multi-label classification. These models are designed to learn non-linear relationships across the engineered biological feature set and produce one probability per target mechanism.

There are also **TabNet experiments** in the repository, which provide an alternate modeling approach for tabular data. Rather than betting everything on one architecture, I explore multiple model families and compare how well they capture the signal.

This is a good idea in general. Biological tabular data can be temperamental, and different model types often succeed on different parts of the problem.

## Validation Strategy

As you might imagine, it would be easy to fool yourself with a model like this if you were not careful about evaluation.

To avoid that, I use **multilabel stratified cross-validation**. This means the data is split into folds in a way that tries to preserve the balance of the target labels across train and validation sets.

That matters because the targets are sparse and unevenly distributed. A naive split can produce misleading results, especially when some mechanisms are rare.

Cross-validation gives a much better estimate of how the model will behave on unseen data.

## Ensembling

Last but not least: ensembling.

Instead of relying on a single model run, I combine predictions from:

- multiple PyTorch runs
- later-stage PyTorch variants
- TabNet experiments
- dedicated ensemble notebooks

This tends to improve stability because different models make different mistakes. When blended well, the final output is usually more reliable than any one component on its own.

If there is one thing I would want someone to take away from this repository, it is probably this: the final performance comes from the **whole workflow**, not from one magic model.

## Repository Layout

Here is what the main notebooks are for:

- [eda.ipynb](/Users/harvardsummer/Downloads/MoA/eda.ipynb): exploratory analysis of the raw features and target behavior
- [pytorch.ipynb](/Users/harvardsummer/Downloads/MoA/pytorch.ipynb): early PyTorch baseline experiments
- [pytorch_iteration_2.ipynb](/Users/harvardsummer/Downloads/MoA/pytorch_iteration_2.ipynb): updated PyTorch iteration
- [pytorch_training_final.ipynb](/Users/harvardsummer/Downloads/MoA/pytorch_training_final.ipynb): later-stage training workflow
- [pytorch_inference_final.ipynb](/Users/harvardsummer/Downloads/MoA/pytorch_inference_final.ipynb): later-stage inference workflow
- [pytorch_best_score_0_01839.ipynb](/Users/harvardsummer/Downloads/MoA/pytorch_best_score_0_01839.ipynb): stronger PyTorch result snapshot
- [tabnet_experiments.ipynb](/Users/harvardsummer/Downloads/MoA/tabnet_experiments.ipynb): TabNet modeling experiments
- [ensemble.ipynb](/Users/harvardsummer/Downloads/MoA/ensemble.ipynb): blending and ensemble logic
- [submission_correlation_analysis.ipynb](/Users/harvardsummer/Downloads/MoA/submission_correlation_analysis.ipynb): submission-correlation analysis

## Tech Stack

This project is built primarily with:

- Python
- Jupyter Notebook
- pandas / NumPy / SciPy
- scikit-learn
- PyTorch
- iterative-stratification
- pytorch-tabnet
- matplotlib / seaborn / Plotly

## Running the Project

To recreate the environment, install the packages listed in [requirements.txt](/Users/harvardsummer/Downloads/MoA/requirements.txt).

Note that several notebooks still reflect the original Kaggle-style file layout, such as references to `../input/lish-moa/...`, so some path cleanup may be needed if you want to run everything locally from scratch.

Also note that this repository is still structured as an experimental notebook workspace rather than a packaged library. If this project were to be extended further, a good next step would be to extract the shared preprocessing, feature engineering, and training code into a `src/` module.

## Final Summary

This project is my machine learning pipeline for turning high-dimensional biological response data into mechanism-of-action predictions. It works by normalizing noisy signals, creating stronger summary features, training multi-label predictive models, and combining multiple model outputs into a more robust final result.

Or, in shorter form:

I take a biological fingerprint and turn it into a structured guess about what a compound is doing.
