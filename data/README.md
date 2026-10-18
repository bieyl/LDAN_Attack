# Public data

The implementation expects user-provided, ImageFolder-compatible data. The experiments in the paper use:

- ImageNet validation images for generator training;
- the NeurIPS 2017 adversarial competition images for transfer evaluation;
- Caltech101 for the auxiliary adversarial-training robustness experiment.

No dataset or generated image is stored in this repository. Configure `train_dir` and `eval_dir` in `configs/default.yaml` for the local layout.

