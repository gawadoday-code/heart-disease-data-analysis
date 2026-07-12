# Analytical Decisions

This file documents the decisions that matter most in an interview.

## 1. Why were cholesterol and resting blood pressure zeros changed to missing?

A cholesterol or resting blood pressure measurement of zero is not clinically plausible. The cholesterol zeros were also concentrated by source, indicating placeholder coding rather than genuine measurements. The original values were preserved in the cleaning audit.

## 2. Why were rows not deleted?

The dataset is small and missingness is source-dependent. Deleting incomplete rows would change the source mix and introduce selection bias. Potential duplicates and outliers were flagged instead of removed without evidence.

## 3. Why use non-parametric tests?

Several numeric variables were skewed and contained valid extreme values. Mann–Whitney U and Spearman correlation were therefore safer descriptive choices than relying only on normal-distribution assumptions.

## 4. Why choose logistic regression?

It performed similarly to more complex models while remaining easier to explain. For a junior portfolio project and a small dataset, interpretability and stability were prioritized over marginal performance gains.

## 5. Why exclude dataset from the model?

Dataset identity is highly associated with disease prevalence and missingness. Including it could allow the model to learn the hospital rather than the patient pattern. Dataset was retained for stratification and external-style validation.

## 6. What is the biggest weakness?

Generalization. Internal performance was strong, but leave-one-dataset-out validation showed that performance was not stable across hospitals.
