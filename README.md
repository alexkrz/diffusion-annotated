# Diffusion Annotated

Collection of resources related to diffusion models

## Setup for DDPM

1. Add git submodule

    ```bash
    git submodule add https://github.com/lucidrains/denoising-diffusion-pytorch.git
    ```

2. Create and activate conda environment

    ```bash
    conda env create -n ddpm -f environment.yml
    conda activate ddpm
    ```

3. Install submodule in editable mode

    ```bash
    cd denoising-diffusion-pytorch
    pip install -e .
    ```

## Tips & Tricks

- Convert jupyter notebook to markdown:

    ```bash
    jupyter nbconvert --to markdown your_notebook.ipynb
    ```

- Use [Marktext editor](https://github.com/marktext/marktext) to convert markdown to .pdf

## Sources

The Jupyter Notebook explaining *Denoising Diffusion Probabilistic Models* and the included images are taken from the following two repositories:

- <https://github.com/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb>
- <https://github.com/huggingface/blog/blob/main/annotated-diffusion.md>

The corresponding blog-post can be accessed here:

- <https://huggingface.co/blog/annotated-diffusion>
