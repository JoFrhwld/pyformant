import librosa
import numpy as np
import multiprocessing as mp

## PRAAT CONSTANTS
WINDOW_MULT = 2
KAISER_COEF = 20.24
CORES = mp.cpu_count()

### These functions subject to change to approximate Praat
def praat2librosa(time_step: float         = 0.01,
                  num_formant: float       = 5.0,
                  max_formant: int         = 5500,
                  window_length: float     = 0.025,
                  pre_emph: float          = 50.0,
                  formant_floor: float     = 90.0,
                  bandwidth_cieling: float = 400.0):
    """
     Translate praat `To formant (burg)` to librosa settings.
    """
    
    target_sr = max_formant * 2

    args_dict = {"target_sr" : target_sr,
                 "alpha" : np.exp(-2 * np.pi * pre_emph * (1/target_sr)),
                 "frame_length" : int((window_length * WINDOW_MULT) * target_sr),
                 "hop_length" : int(time_step * target_sr),
                 "order" : int(num_formant * 2)}

    return(args_dict)


def praat_window(frame_length: int, **kwargs):
    """
        return default praat window
    """
    window = librosa.filters.get_window(window = ("kaiser", KAISER_COEF), 
                                        Nx = frame_length)
    return(window)
####

def resample(wav: np.ndarray,
             orig_sr: int,
             target_sr: int,
             **kwargs):
    """
     resample using librosa
    """

    out = librosa.resample(wav, orig_sr = orig_sr, target_sr = target_sr)
    return(out)

def preemph(wav: np.ndarray, 
            alpha: float, 
            **kwargs):
    """
    preemphasis
    """

    out = librosa.effects.preemphasis(wav, coef = alpha)
    return(out)

def framing(wav: np.ndarray,
            frame_length: int,
            hop_length: int,
            **kwargs):
    """
        split wav up into analusis frames
    """

    framed = librosa.util.frame(wav, 
                               frame_length=frame_length, 
                               hop_length=hop_length)
    return(framed)

def frame_timepoint(frames: np.ndarray,
                    target_sr : int,
                    hop_length: int,
                    frame_length: int,
                    **kwargs):
    """
      return the time point of the center of frames
    """

    halfpoint = ((frame_length-1)/2)/target_sr
    frame_time_left = librosa.frames_to_time(frames, 
                                             sr = target_sr, 
                                             hop_length=hop_length)
    frame_time_center = frame_time_left + halfpoint

    return(frame_time_center)

def window_frames(frames: np.ndarray,
                  window: np.ndarray,
                  **kwargs):
    """
        apply windowing function to frames
    """

    windowed = frames * window[:,np.newaxis]
    return(windowed)

def get_lpcs(frames: np.ndarray,
             order: int,
             axis = 0,
             **kwargs):
    """
        get LPC coefficients
    """
    lpcs = librosa.lpc(y = frames, order = order, axis = axis)

    return(lpcs)

def get_roots(lpcs: np.ndarray):
    """
        get the roots of a set of lpc coefficients
    """
    roots= np.apply_along_axis(np.roots, 0, lpcs)
    return(roots)

def get_angles(roots: np.ndarray):
    """
        get the angles
    """

    angles = np.apply_along_axis(np.angle, 0, roots)
    return(angles)

def angle_to_freq(angles: np.ndarray,
                  target_sr: int,
                  **kwargs):
    """
        convert angles to frequencies
    """

    def angle1(angle, target_sr):
        """
            single angle to sr
        """

        F = angle * (target_sr /(2 * np.pi))
        return(F)
    
    formants = np.apply_along_axis(angle1, 0, angles, target_sr = target_sr)
    return(formants)

def get_bandwidths(roots: np.ndarray,
                   target_sr: int,
                   **kwargs):
    """
        get bandwidths
    """
    def bw1(root, target_sr):
        """
            get one bandwidth
        """
        bw = (-1/2)*(target_sr/(2*np.pi))*np.log(abs(root))
        return(bw)
    
    bws = np.apply_along_axis(bw1, 0, roots, target_sr = target_sr)
    return(bws)

def filter_formant1d(formants: np.ndarray,
                     bandwidths: np.ndarray):
    """
        with 1d arrays, filter
    """
    out_formants = formants[(formants > 90) & (bandwidths < 400)]
    #out_bandwidths = bandwidths[(formants > 90) & (bandwidths < 400)]
    return(out_formants)

def filter_formants(formants: np.ndarray,
                    bandwidths: np.ndarray,
                    formant_floor: float = 90,
                    bandwidth_cieling = 400):
    """
    filter and sort formants
    """

    formants[~((formants > formant_floor) & (bandwidths < bandwidth_cieling))] = None
    bandwidths[np.isnan(formants)] = None

    return(formants, bandwidths)