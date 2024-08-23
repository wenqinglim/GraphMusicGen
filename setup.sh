conda create --prefix ~/GraphMusicGen/envs python=3.7
# conda activate /home/jovyan/GraphMusicGen/envs
source activate /home/jovyan/GraphMusicGen/envs

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html


# To add conda env as jupyter kernel:
python -m ipykernel install --user --name=graphmugen

# Pre-processing command
python3 preprocess_pop909.py POP909 POP909_structure preprocessed_909 --n_workers=6

# Training command
python3 train.py preprocessed_909 models training.json --model_name POP8 --use_gpu

# Generation command
python3 generate.py models/POP8/ music/ --n 10 --n_loops 4

