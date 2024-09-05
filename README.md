# Diffusion Annotated

Collection of resources related to diffusion models

## Setup

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
