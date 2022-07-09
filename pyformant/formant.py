import librosa
import numpy as np

def praat2librosa(time_step: float     = 0.01,
                  num_formant: float   = 5.0,
                  max_formant: int     = 5500,
                  window_length: float = 0.025,
                  pre_emph: float      = 50.0):
    """
     Translate praat `To formant (burg)` to librosa settings.
    """
    
    target_sr = max_formant * 2

    args_dict = {"target_sr" : target_sr,
                 "alpha" : np.exp(-2 * np.pi * pre_emph * (1/target_sr)),
                 "frame_length" : int((window_length * 2) * target_sr),
                 "hop_length" : int(time_step * target_sr)}

    return(args_dict)


