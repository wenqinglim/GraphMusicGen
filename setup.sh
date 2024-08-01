conda create --prefix ~/GraphMusicGen/envs python=3.7
# conda activate /home/jovyan/GraphMusicGen/envs
source activate /home/jovyan/GraphMusicGen/envs


# To add conda env as jupyter kernel:
python -m ipykernel install --user --name=graphmugen
