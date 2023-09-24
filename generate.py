import argparse
import os
import json
import time

import torch
import numpy as np
import os
from torch.utils.data import Subset
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import random
from matplotlib import pyplot as plt
from utils import set_seed

from data import MIDIDataset, graph_from_tensor, graph_from_tensor_torch
from model import VAE
from utils import plot_struct
from utils import plot_pianoroll, midi_from_muspy
from train import VAETrainer

import matplotlib as mpl
from utils import mtp_from_logits, muspy_from_mtp


def generate_music(vae, z, s_cond=None, s_tensor_cond=None):
    
    # Get structure and content logits
    _, c_logits, s_tensor_out = vae.decoder(z, s_cond)
    
    s_tensor = s_tensor_cond if s_tensor_cond != None else s_tensor_out
    
    # Build (n_batches x n_bars x n_tracks x n_timesteps x Sigma x d_token)
    # multitrack pianoroll tensor containing logits for each activation and
    # hard silences elsewhere
    mtp = mtp_from_logits(c_logits, s_tensor)
    
    return mtp, s_tensor


def save(mtp, dir, resolution, s_tensor=None, track_data=None, n_loops=1):
    
    track_data = ([('Drums', -1), ('Bass', 34), ('Guitar', 1), ('Strings', 83)]
                  if track_data == None else track_data)
    
    # Clear matplotlib cache (this avoids formatting problems with first plot)
    plt.clf()

    # Iterate over the generated n-bar sequences
    for i in range(mtp.size(0)):
        
        # Create the directory if it does not exist
        save_dir = os.path.join(dir, str(i))
        os.makedirs(save_dir, exist_ok=True)

        print("Saving MIDI sequence {}...".format(str(i+1)))
        
        # Generate muspy song from multitrack pianoroll, then midi from muspy
        # and save
        muspy_song = muspy_from_mtp(mtp[i], track_data, resolution)
        midi_from_muspy(muspy_song, save_dir, name='music')
        
        # Plot the pianoroll associated to the sequence
        preset = 'full'
        with mpl.rc_context({'lines.linewidth': 4, 
                             'axes.linewidth': 4, 'font.size': 34}):
            plot_pianoroll(muspy_song, save_dir, name='pianoroll',
                           figsize=(20, 10), fformat='png', preset=preset)
        
        # Plot structure_tensor if present
        if s_tensor is not None:
            s_curr = s_tensor[i]
            s_curr = s_curr.permute(1, 0, 2)
            s_curr = s_curr.reshape(s_curr.shape[0], -1)
            with mpl.rc_context({'lines.linewidth': 1, 
                                 'axes.linewidth': 1, 'font.size': 14}):
                plot_struct(s_curr.cpu(), name='structure', 
                            save_dir=save_dir, figsize=(12, 3))

        if n_loops > 1:
            # Generate extended sequence
            print("Saving extended MIDI sequence " \
                  "{} with {} loops...".format(str(i+1), n_loops))
            extended = mtp[i].repeat(n_loops, 1, 1, 1, 1)
            extended = muspy_from_mtp(extended, track_data, resolution)
            midi_from_muspy(extended, save_dir, name='extended')
        
        print()
    

def generate_z(bs, d_model, device):
    shape = (bs, d_model)

    z_norm = torch.normal(
        torch.zeros(shape, device=device),
        torch.ones(shape, device=device)
    )
    
    return z_norm
    

def load_model(model_dir, device):

    checkpoint = torch.load(os.path.join(model_dir, 'checkpoint'), 
                            map_location='cpu')
    params = torch.load(os.path.join(model_dir, 'params'), map_location='cpu')
    
    state_dict = checkpoint['model_state_dict']
    
    model = VAE(**params['model'], device=device).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, params


def main():
    
    parser = argparse.ArgumentParser(
        description='Generates MIDI music with a trained model.'
    )
    parser.add_argument(
        'model_dir', 
        type=str, help='Directory of the model.'
    )
    parser.add_argument(
        'output_dir', 
        type=str, 
        help='Directory to save the generated MIDI files.'
    )
    parser.add_argument(
        '--n', 
        type=int,
        default=5, 
        help='Number of sequences to be generated. Default is 5.'
    )
    parser.add_argument(
        '--n_loops', 
        type=int,
        default=1, 
        help="If greater than 1, outputs an additional MIDI file containing " \
                "the sequence looped n_loops times."
    )
    parser.add_argument(
        '--s_file', 
        type=str, 
        help='Path to the file containing the binary structure tensor.'
    )
    parser.add_argument(
        '--use_gpu',
        action='store_true',
        default=False, 
        help='Flag to enable or disable GPU usage. Default is False.'
    )
    parser.add_argument(
        '--gpu_id', 
        type=int, 
        default='0', 
        help='Index of the GPU to be used. Default is 0.'
    )
    parser.add_argument(
        '--seed', 
        type=int
    )
    
    args = parser.parse_args()
    
    if args.seed is not None:
        set_seed(args.seed)
        
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    if args.use_gpu:
        torch.cuda.set_device(args.gpu_id)
        
    print("------------------------------------")
    print("Loading the model on {} device...".format(device))
    
    model, params = load_model(args.model_dir, device)

    d_model = params['model']['d']
    n_bars = params['model']['n_bars']
    n_tracks = params['model']['n_tracks']
    resolution = params['model']['resolution']
    n_timesteps = 4*resolution
    output_dir = args.output_dir

    bs = args.n
    track_data = [('Drums', -1), ('Bass', 34), ('Guitar', 1), ('Strings', 83)]

    s, s_tensor = None, None

    if args.s_file is not None:
        
        print("Loading the structure tensor " \
               "from {}...".format(args.model_dir))
        
        # Load structure tensor from file
        with open(args.s_file, 'r') as f:
            s_tensor = json.load(f)
        
        s_tensor = torch.tensor(s_tensor)
        
        # Check structure dimensions
        dims = list(s_tensor.size())
        expected = [n_bars, n_tracks, n_timesteps]
        if dims != expected:
            raise ValueError(f"Loaded tensor dimensions {dims} " \
                             f"do not match expected dimensions {expected}")
        
        s_tensor = s_tensor.bool()
        s_tensor = s_tensor.unsqueeze(0).repeat(bs, 1, 1, 1)
        s = model.decoder._structure_from_binary(s_tensor)
    
    print()
    print("Generating z...")
    z = generate_z(bs, d_model, device)
    
    print("Generating music with the model...")
    s_t = time.time()
    mtp, s_tensor = generate_music(model, z, s, s_tensor)
    print("Inference time: {:.3f} s".format(time.time()-s_t))
    
    print()
    print("Saving MIDI files in {}...".format(output_dir))
    save(mtp, output_dir, resolution, s_tensor, track_data, args.n_loops)
    print("Finished saving MIDI files.")


if __name__ == '__main__':
    main()
