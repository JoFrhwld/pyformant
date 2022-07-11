import librosa
import numpy as np
import pandas as pd

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
                    bandwidth_ceiling = 400):
    """
    filter and sort formants
    """

    formants[~((formants > formant_floor) & (bandwidths < bandwidth_ceiling))] = None
    bandwidths[np.isnan(formants)] = None

    return(formants, bandwidths)

class SamplingRateException(Exception):
    """
    For problems with sampling rates
    """
    def __init__(self, message):
        super().__init__(message)

class OrderException(Exception):
    """
    For problems with lpc orders
    """
    def __init__(self, message):
        super().__init__(message)

class VowelLike:
    """
        Wrapping it all up
    """
    def __init__(self,
                 path: str = None,
                 wav: np.ndarray = None,
                 load_sr: int = None,
                 start_s: float = 0.0,
                 end_s: float = None,
                 max_formant: int = 5500,
                 praat_preemph: float = 50.0,
                 preemph: float = None,
                 window_len_s: float = 0.05,
                 step_size_s: float = 0.01,
                 window = "praat",
                 n_formants: float = 5,
                 formant_floor: float = 90.0,
                 bandwidth_ceiling: float = 400
                 ):

        for k in locals():
            setattr(self, k, None)
        self.harmonize_args(locals())

        if self.path is not None:
            self.load(self.path)
        elif self.wav is not None:
            if self.load_sr is None:
                raise SamplingRateException("wav provided with no sampling rate")           
            self.end_s = self.wav.shape[0]/self.load_sr
            self.dur = self.end_s 

        if self.preemph is not None:
            if self.preemph > 1:
                print("preemph > 1, did you mean to set praat_preemph with a Hz like value?")
        else:
            if self.target_sr is not None:
                self.preemph = np.exp(-2 * np.pi * self.praat_preemph * (1/self.target_sr))
        

    def __repr__(self):
        info_message = "audio:"
        if self.wav is not None:
            if self.path is not None:
                info_message += f"\t- loaded from path {self.path}\n"
            if self.start_s != 0:
                info_message += f"\t- slice from {self.start_s:.3f} (s)"
                if self.end_s is not None:
                    info_message += f" to {self.end_s:.3f} (s)\n"
                else:
                    info_message += "\n"
            info_message += f"\t- {self.wav.shape[0]} samples at {self.load_sr} sampling rate ({self.dur:.3f} seconds)\n"
            if self.target_sr is not None:
                info_message += f"\t- resampled to {self.target_sr} for max_formant {self.max_formant}\n"
            if self.praat_preemph is not None:
                info_message += f"\t- preemphasis added from {self.praat_preemph:.2f} Hz\n"
            if self.window_len_s is not None and\
              self.step_size_s is not None and self.wav_framed is not None:
                 info_message += f"\t- split into {self.wav_framed.shape[1]} frames {self.window_len_s} s long with {self.step_size_s} s step size\n"
        else:
            info_message += "No wav file loaded\n"
        if self.n_formants is not None:
            info_message += "LPCs"
            info_message += f"\t- LPC order {self.order} to get {self.n_formants} formants\n"
        else:
            info_message += "No LPC settings"

        return(info_message)

    def harmonize_args(self, args):
        """
        harmonize args from this function to self attributes
        """
        for k in args:
            if args[k] is None:
                args[k] = self.__dict__[k]
            setattr(self, k, args[k])

    def load(self, 
             path: str = None,
             load_sr: float = None,
             start_s: float = None,
             end_s: float = None):
        """
        
        """

        self.harmonize_args(locals())

        if self.load_sr is None:
            self.load_sr = librosa.get_samplerate(self.path)
        if self.end_s is not None:
            self.dur = self.end_s - self.start_s
        else:
            self.end_s = librosa.get_duration(filename=self.path)
            self.dur = self.end_s - self.start_s

        self.wav, _= librosa.load(path, sr=self.load_sr, offset=self.start_s, duration=self.dur)
    
    @property
    def target_sr(self):
        if self.max_formant is None:
            return(None)
        else:
            return(self.max_formant * 2)

    @property        
    def wav_r(self):
        """
            resample to 2x nyquist frequency
        """

        if self.wav is not None:
            if self.target_sr is not None:
                if self.target_sr < self.load_sr:
                    return(librosa.resample(self.wav, orig_sr=self.load_sr, target_sr = self.target_sr))
                else:
                    raise SamplingRateException("loaded sampling rate too low for given max_formant")
            else:
                return(None)
        else:
            return(None)
    
    @property
    def wav_preemph(self):
        """
            add preemphasis
        """

        if self.wav_r is not None and \
           self.preemph is not None:
             return(librosa.effects.preemphasis(self.wav_r, coef = self.preemph))
        else:
            rerturn(None)

    
    @property
    def frame_length(self):
        """
        get window length in frames
        """
        if self.window_len_s is not None and \
           self.target_sr is not None:
             return(int(self.window_len_s * self.target_sr))
        else:
            return(None)
    
    @property
    def hop_length(self):
        """
        get hop length
        """
        if self.step_size_s is not None and\
           self.target_sr is not None:
             return(int(self.step_size_s * self.target_sr))
        else:
            return(None)
    
    @property
    def wav_framed(self):
        """
         split wav_preemph into analysis frames
        """
        if self.wav_preemph is not None and \
           self.frame_length is not None and \
           self.hop_length is not None:
             return(librosa.util.frame(self.wav_preemph, 
                                       frame_length=self.frame_length, 
                                       hop_length=self.hop_length))
        else:
            return(None)
    
    @property
    def window_array(self):
        """
        get window function
        """
        if self.window is not None:
            if self.window == "praat":
                return(praat_window(frame_length=self.frame_length))
            else:
                return(librosa.filters.get_window(window = self.window, 
                                                  Nx = self.frame_length))
        else:
            return(None)

    @property
    def frame_time(self):
        """
        return frame times
        """

        if self.wav_framed is not None:
            halfpoint = ((self.frame_length-1)/2)/self.target_sr
            return(librosa.frames_to_time(range(self.wav_framed.shape[1]), 
                                          sr = self.target_sr, 
                                          hop_length=self.hop_length) + halfpoint)
    
    @property
    def wav_windowed(self):
        """
        apply windowing function
        """
        if self.window_array is not None and self.wav_preemph is not None:
            return(self.wav_framed * self.window_array[:,np.newaxis])
        else:
            return(None)
    
    @property
    def order(self):
        """
        get lpc order
        """
        if self.n_formants is not None:
            if (self.n_formants * 2) % 1 != 0:
                raise OrderException("n_formant*2 must be whole number")
            else:
                return(int(self.n_formants * 2))
        else:
            return(None)

    @property
    def lpcs(self):
        """
        get lpcs
        """

        if self.wav_windowed is not None and \
           self.order is not None:
             return(librosa.lpc(y = self.wav_windowed, order = self.order, axis = 0))
        else:
            return(None)

    @property
    def roots(self):
        """
        get roots of LPCs
        """

        if self.lpcs is not None:
            return(np.apply_along_axis(np.roots, 0, self.lpcs))
        else:
            return(None)
    
    @property
    def angles(self):
        """
        get angles of roots
        """

        if self.roots is not None:
            return(np.apply_along_axis(np.angle, 0, self.roots))
        else:
            return(None)
    
    @property
    def frequencies(self):
        """
        convert angles to frequencies
        """

        def angle1(angle, target_sr):
            """
            func for one angle
            """
            F = angle * (target_sr /(2 * np.pi))
            return(F)
        
        if self.angles is not None:
            return(np.apply_along_axis(angle1, 0, self.angles, target_sr = self.target_sr))
        else:
            return(None)

    @property
    def bws(self):

        def bw1(root, target_sr):
            """
                get one bandwidth
            """
            bw = (-1/2)*(target_sr/(2*np.pi))*np.log(abs(root))
            return(bw)
        
        if self.roots is not None:
            return(np.apply_along_axis(bw1, 0, self.roots, target_sr = self.target_sr))
        else:
            return(None)
    
    @property
    def positive_frequencies(self):
        """
        get list of formant arrays
        """
        if self.frequencies is not None:
            out_freq = self.frequencies
            out_freq[~(self.frequencies > self.formant_floor)] = None
            out_freq[~(self.bws < self.bandwidth_ceiling)] = None
            return(out_freq)
        else:
            return(None)
    
    @property
    def formant_idx(self):
        """
        get formant indices
        """
        if self.positive_frequencies is not None:
            return(np.argsort(self.positive_frequencies, axis = 0))
        else:
            return(None)
    
    @property
    def formants_array(self):
        """
        array of formants
        """
        if self.positive_frequencies is not None:
            max = int(round(self.n_formants))
            return(np.take_along_axis(self.positive_frequencies, 
                                      self.formant_idx, 
                                      axis = 0)[0:max,:])
        else:
            return(None)
    
    @property
    def bandwidths_array(self):
        """
        array of bandwidths
        """
        if self.positive_frequencies is not None:
            max = int(round(self.n_formants))
            return(np.take_along_axis(self.bws, 
                                      self.formant_idx, 
                                      axis = 0)[0:max,:])
        else:
            return(None)

    @property
    def formant_df(self):
        """
        generate a formant table
        """
        max = int(round(self.n_formants))
        if self.formants_array is not None and \
           self.bandwidths_array is not None and \
           self.frame_time is not None:
            f_tab = pd.DataFrame(self.formants_array.T)
            f_tab.columns = ["F" + repr(x+1) for x in range(f_tab.shape[1])]
            b_tab = pd.DataFrame(self.bandwidths_array.T)
            b_tab.columns = ["B" + repr(x+1) for x in range(b_tab.shape[1])]

            out_df = pd.concat([f_tab, b_tab], axis = 1)
            out_df.insert(0, "time_s", self.frame_time)
            return(out_df)
        else:
            return(None)
