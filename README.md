# GraphMuGen

This is the repository for [Hierarchical Symbolic Pop Music Generation with Graph Neural Networks](https://arxiv.org/abs/2409.08155), 
and builds on [Polyphemus](https://github.com/EmanueleCosenza/polyphemus/)


## Setting Up
1. **Create an Environment:**
   Use conda to create a Python 3.10 environment and activate it:
   ```sh
   conda create -n venv python=3.10
   conda activate venv
   ```
   Alternatively, you can install Python 3.10 and create a new virtual environment using `venv`.

2. **Clone the Repository:**
   ```sh
   git clone https://github.com/wenqinglim/GraphMusicGen
   ```
   
4. **Install the Required Python Packages:**
   Navigate to the directory where the repository is cloned and install the required packages from `requirements.txt`:
   ```sh
   pip3 install -r requirements.txt
   ```
5. **Prepare Your Environment for Audio Generation (Recommended):**
    Refer to `README_Polyphemus.md` for [`fluidsynth`](https://github.com/FluidSynth/fluidsynth/wiki) and a [SoundFont](https://github.com/FluidSynth/fluidsynth/wiki/SoundFont) setup details.

6. **Download the Trained Models:**


Helper command line scripts are in `setup.sh`
WARNING: You might face dependency mismatches, these are currently unresolved. 

## Key Scripts and Notebooks
The following are the main scripts/notebooks for each section:
### Phrase Generation
Generate 4-bar phrases
- `preprocess_pop909.py`
- `train.py`
- `generate.py`

### Song Structure generation
Generate full song structure
- `preprocess_structure.py`
- `train_structure.py`
- `generate_structure.py`

### Interpolation
Interpolate between phrases
- `interpolation.ipynb`

### Evaluation
Evaluate models and plot metrics
- `eval_notebooks/`
