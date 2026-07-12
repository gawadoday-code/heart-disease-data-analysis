# Portfolio Presentation Notes

## Two-minute project story

I used a multi-source heart disease dataset to answer two questions: which clinical features are associated with disease, and whether a simple predictive model generalizes across hospitals.

The hardest issue was not the algorithm—it was source heterogeneity. Missingness and disease prevalence differed substantially across hospitals. I therefore preserved the source variable for stratified analysis, avoided deleting incomplete rows, and tested transportability by holding out one hospital at a time.

Logistic regression performed similarly to more complex models and was selected because it was easier to explain. Internal performance was strong, but performance on unseen hospitals was less stable. My final conclusion was that the model is appropriate as a portfolio baseline, not as a clinical tool.

## Questions to prepare for

- Why did you avoid deleting incomplete rows?
- Why did you choose Mann–Whitney U?
- Why was dataset excluded from model inputs but used in validation?
- Why did you select logistic regression?
- What would be required before clinical deployment?
