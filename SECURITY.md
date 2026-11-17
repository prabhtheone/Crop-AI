# Security and responsible use

Crop AI is a learning/research prototype. Its output is not a certified agricultural inspection, food-safety, or commercial grading decision.

## Data and credentials

- Do not commit API keys, credentials, private images, or personal information.
- Keep private datasets out of Git.
- Keep trained model files in Git LFS.
- Review staged files before committing.

## Model limitations

- Quality grading is currently available for Guava only.
- Confidence values are model scores and are not calibrated probabilities.
- The reported evaluation results are not guarantees of performance on new farms, varieties, cameras, or lighting conditions.
- A human should review outputs before using them for a commercial or financial decision.

## Reporting

Do not publish credentials or private data in an issue. Report security-sensitive problems privately to the repository owner through GitHub.
