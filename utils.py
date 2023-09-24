from matplotlib import pyplot as plt
import numpy as np
import copy
import muspy
import os
from scipy.special import softmax
import torch
import random

from constants import PitchToken, DurationToken
import constants


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# Builds multitrack pianoroll (mtp) from content tensor containing logits and 
# structure binary tensor
def mtp_from_logits(c_logits, s_tensor):
    
    mtp = torch.zeros((s_tensor.size(0), s_tensor.size(1), s_tensor.size(2), 
                       s_tensor.size(3), c_logits.size(-2), c_logits.size(-1)),
                       device=c_logits.device, dtype=c_logits.dtype)

    size = mtp.size()
    mtp = mtp.view(-1, mtp.size(-2), mtp.size(-1))
    silence = torch.zeros((mtp.size(-2), mtp.size(-1)),
                          device=c_logits.device, dtype=c_logits.dtype)
    
    # Create silences with pitch EOS and PAD tokens
    silence[0, 129] = 1.
    silence[1:, 130] = 1.

    # Fill the multitrack pianoroll
    mtp[s_tensor.bool().view(-1)] = c_logits
    mtp[torch.logical_not(s_tensor.bool().view(-1))] = silence
    mtp = mtp.view(size)
    
    return mtp


def muspy_from_mtp(mtp, track_data, resolution):
    
    # Collapse bars dimension
    mtp = mtp.permute(1, 0, 2, 3, 4)
    size = (mtp.shape[0], -1, mtp.shape[3], mtp.shape[4])
    mtp = mtp.reshape(*size)
    
    tracks = []
    
    for tr in range(mtp.size(0)):
        
        notes = []
        
        for ts in range(mtp.size(1)):
            for note in range(mtp.size(2)):
                
                # Compute pitch and duration values
                pitch = mtp[tr, ts, note, :constants.N_PITCH_TOKENS]
                dur = mtp[tr, ts, note, constants.N_PITCH_TOKENS:]
                pitch, dur = torch.argmax(pitch), torch.argmax(dur)

                if (pitch == PitchToken.EOS.value or 
                    pitch == PitchToken.PAD.value or
                    dur == DurationToken.EOS.value or 
                    dur == DurationToken.PAD.value):
                    # This chord contains no additional notes, go to next chord
                    break
                
                if (pitch == PitchToken.SOS.value or
                    pitch == PitchToken.SOS.value):
                    continue
                
                # Remapping [0, 95] to [1, 96] real duration values
                dur = dur + 1
                # Do not sustain notes beyond sequence limit
                dur = min(dur.item(), mtp.size(1)-ts)
                
                notes.append(muspy.Note(ts, pitch.item(), dur, 64))
        
        
        if track_data[tr][0] == 'Drums':
            track = muspy.Track(name='Drums', is_drum=True, notes=copy.deepcopy(notes))
        else:
            track = muspy.Track(name=track_data[tr][0], 
                                program=track_data[tr][1],
                                notes=copy.deepcopy(notes))
        tracks.append(track)
    
    meta = muspy.Metadata(title='prova')
    music = muspy.Music(tracks=tracks, metadata=meta, resolution=resolution)
    
    return music


def plot_pianoroll(music, save_dir=None, name=None, figsize=(10, 10),
                   fformat="png", xticklabel='on', preset='full', **kwargs):

    fig, axs_ = plt.subplots(4, sharex=True, figsize=figsize)
    fig.subplots_adjust(hspace=0)
    axs = axs_.tolist()
    muspy.show_pianoroll(music=music, yticklabel='off',
                         xticklabel=xticklabel, grid_axis='off',
                         axs=axs, preset=preset, **kwargs)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, name+"."+fformat), format=fformat, dpi=200)
        
        
def plot_struct(s, save_dir=None, name=None, figsize=(10, 10), fformat="svg", n_bars=2):
    
    plt.figure(figsize=figsize)
    plt.pcolormesh(s, edgecolors='k', linewidth=1)
    ax = plt.gca()
    
    plt.xticks(range(0, s.shape[1], 8), range(1, n_bars*4+1))
    plt.yticks(range(0, 4), ['Drums', 'Bass', 'Guitar', 'Strings'])
    
    ax.invert_yaxis()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, name+"."+fformat), format=fformat, dpi=200)
        

def midi_from_muspy(music, save_dir, name):
    muspy.write_midi(os.path.join(save_dir, name+".mid"), music)
    