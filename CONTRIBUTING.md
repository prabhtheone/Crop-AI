# Contributing

Thanks for taking an interest in Crop AI.

## Development setup

1. Install Python and Git.
2. Install Git LFS and run `git lfs pull` after cloning.
3. Create a virtual environment and install `requirements.txt`.
4. Run the checks below before opening a pull request.

## Checks

```bash
python -m py_compile src/predict.py
python -m unittest discover -s tests -v
python -m json.tool notebook/Crop_AI_Inference_Demo.ipynb > /dev/null
```

The unit tests use fake model outputs. They verify prediction handling, class-count checks, and missing Git LFS objects; they do not evaluate the trained models.

## Pull requests

Keep changes focused and explain:

- what changed and why;
- how the change was tested;
- whether model behaviour, class order, input size, or data requirements changed;
- any new limitations or untested cases.

Do not commit datasets, credentials, virtual environments, or generated files. Keep trained `.keras` files under Git LFS. Do not describe validation accuracy as test accuracy or claim real-world performance without a documented evaluation.

Use fictional or public-permission sample images when testing.
