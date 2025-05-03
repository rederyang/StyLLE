# StyLLE

Style Learning and Latent Editing (StyLLE) is a method for stylizing autoregressive generation of decoder-only transformer models, based on the paper [DRESSing Up LLM: Efficient Stylized Question-Answering via Style Subspace Editing](https://arxiv.org/abs/2501.14371).

## Environment Setup

```bash
conda create -n stylle python=3.12.3
conda activate stylle
pip install -r requirements.txt
```

## Run

```bash
bash run.sh <dataset> <model_dir> <assets_dir>
```

-   `<dataset>`: Specifies the dataset to use (e.g., "DRC", "Shakespeare").
-   `<model_dir>`: Specifies the directory containing the pre-trained model.
-   `<assets_dir>`: Specifies the directory for generated assets specific to the model and dataset.

## Experiment Logs

All experiment logs are available [here](https://docs.google.com/spreadsheets/d/1v61RFgWgZ46yZCdcOApYErBoQdCuIgrxbSF7pM1BSyY/edit?usp=sharing).