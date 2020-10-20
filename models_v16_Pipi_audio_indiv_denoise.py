import numpy as np
import os
import sys
import types
import pywt

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import requests, io
from scipy.io import wavfile
import mywavfile
#from features import mfcc, logfbank
#https://python-speech-features.readthedocs.io/en/latest/
from python_speech_features import mfcc
from python_speech_features import logfbank


from numpy import savetxt #jonas test
import write_op as wo

import time
start = time.time()

save_individual_results = True
save_summary_result = True

from collections import Iterable

def flatten(lis):
    '''convert nested list into one dimensional list'''
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item

def spectral_centroid(x, samplerate):
    ''' source: https://stackoverflow.com/questions/24354279/python-spectral-centroid-for-a-wav-file'''

    magnitudes = np.abs(np.fft.rfft(x)) # magnitudes of positive frequencies
    length = len(x)
    freqs = np.abs(np.fft.fftfreq(length, 1.0/samplerate)[:length//2+1]) # positive frequencies
    return np.sum(magnitudes*freqs) / np.sum(magnitudes) # return weighted mean


def pad(array, reference_shape, offsets=None):
    """
    array: Array to be padded
    reference_shape: tuple of size of narray to create
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    will throw a ValueError if offsets is too big and the reference_shape cannot handle the offsets
    """

    if not offsets:
        offsets = np.zeros(array.ndim, dtype=np.int32)
        #offsets = np.zeros(array.ndim, dtype=object)

    # Create an array of zeros with the reference shape
    result = np.zeros(reference_shape, dtype=np.float32)
    #result = np.zeros(reference_shape, dtype=object)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offsets[dim], offsets[dim] + array.shape[dim]) for dim in range(array.ndim)]
    # Insert the array in the result at the specified offsets
    result[insertHere] = array
    return result

def _denoise(spec):
    """
    Perform denoising.
    """
    me2 = np.mean(spec)
    spec = spec - me2
    # remove anything below 0
    spec.clip(min=0, out=spec)
    return spec


#function to get the audio files
def get_audio_files(ip_dir):
    matches = []
    for root, dirnames, filenames in os.walk(ip_dir):
        for filename in filenames:
            if filename.lower().endswith('.wav'):
                matches.append(os.path.join(root, filename))
    return matches


def read_audio(file_name):
    # try to read in audio file
    try:
        #samp_rate_orig, audio = mywavfile.read(file_name)
        samp_rate_orig, audio = wavfile.read(file_name)

        #coeff, freqs = pywt.cwt(file_name, scales, 'morl')
    except:
        print('  Error reading file')
        return True, None, None, None, None

    # convert to mono if stereo
    if len(audio.shape) == 2:
        print('  Warning: stereo file. Just taking left channel.')
        audio = audio[:, 0]
    file_dur = audio.shape[0] / float(samp_rate_orig)
    print('  dur', round(file_dur,3), '(secs) , fs', samp_rate_orig)


    # original model is trained on time expanded data
    samp_rate = samp_rate_orig
    audio = _denoise(audio)
    #audio = audio / (2.**15) #normalize the values
    #source: https://github.com/JonasMok/Python-Machine-Learning-Cookbook/blob/master/Chapter07/extract_mfcc.py
    #mfcc_features = mfcc(audio, samp_rate_orig)
    #filterbank_features = logfbank(audio, samp_rate_orig)
    #len_mfcc = mfcc_features.shape
    #len_filter = filterbank_features.shape


    spec_centroid = spectral_centroid(audio, samp_rate_orig)

    #return False, audio, file_dur, samp_rate, samp_rate_orig, mfcc_features, filterbank_features, len_filter, len_mfcc, spec_centroid
    return False, audio, file_dur, samp_rate, samp_rate_orig, spec_centroid
    #return False, audio, file_dur, samp_rate, samp_rate_orig, spec_centroid

#--------------------------------------------------------------------------------------------------------------------------------------

def printar (audio_files, data_dir):
    print('Processing        ', len(audio_files), 'files')
    print('Input directory   ', data_dir)


# loop through audio files
def resultado(audio_files,data_dir):
    results = []
    for file_cnt, file_name in enumerate(audio_files):
        file_name_basename = file_name[len(data_dir)+1:]
        print('\n', file_cnt+1, 'of', len(audio_files), '\t', file_name_basename)

        #read_fail, audio, file_dur, samp_rate, samp_rate_orig, spec_centroid = read_audio(file_name)
        read_fail, audio, file_dur, samp_rate, samp_rate_orig, spec_centroid = read_audio(file_name)

        if read_fail:
            continue

        #res = {'label': data_dir,'filename':file_name_basename, 'sample_rate':samp_rate_orig, 'spec_centroid':spec_centroid} #'coefficient':coeff, 'frequency':freqs,
        #res = {'label': data_dir,'filename':file_name_basename, 'sample_rate':samp_rate_orig, 'mfcc': mfcc_features, 'filterbank':filterbank_features, 'len_mfcc':len_mfcc, 'len_filter': len_filter, 'spec_centroid':spec_centroid}
        res = {'label': data_dir,'filename':file_name_basename, 'sample_rate':samp_rate_orig, 'spec_centroid':spec_centroid}
        #res = {'data_dir':data_dir,'filename':file_name_basename, 'sample_rate':samp_rate_orig}
        results.append(res)

    return results

#-----------------------------------------------------------------------------------------------------------------------------------------------------------

# this is the path to your audio files

#data_dir = 'test'
#data_dir2 = 'test_2'
#data_dir3='test_3'

data_dir_control = 'Control 0006246'
data_dir_ground_1 = 'T1 Ground 0006199'
data_dir_nacelle_1 = 'T1 Nacelle 0006325'
data_dir_ground_5 = 'T5 Ground 0006323'
data_dir_nacelle_5 = 'T5 Nacelle take 2'
data_dir_ground_9 = 'T9 Ground 0006331'
data_dir_nacelle_9 = 'T9 Nacelle 0006364'



audio_files = get_audio_files(data_dir_control)
printar(audio_files, data_dir_control)
result_1=resultado(audio_files,data_dir_control)

audio_files_2 = get_audio_files(data_dir_ground_1)
printar(audio_files_2, data_dir_ground_1)
result_2=resultado(audio_files_2,data_dir_ground_1)

audio_files_3 = get_audio_files(data_dir_nacelle_1)
printar(audio_files_3, data_dir_nacelle_1)
result_3=resultado(audio_files_3,data_dir_nacelle_1)

audio_files_4 = get_audio_files(data_dir_ground_5)
printar(audio_files_4, data_dir_ground_5)
result_4=resultado(audio_files_4,data_dir_ground_5)

audio_files_5 = get_audio_files(data_dir_nacelle_5)
printar(audio_files_5, data_dir_nacelle_5)
result_5=resultado(audio_files_5,data_dir_nacelle_5)

audio_files_6 = get_audio_files(data_dir_ground_9)
printar(audio_files_6, data_dir_ground_9)
result_6=resultado(audio_files_6,data_dir_ground_9)

audio_files_7 = get_audio_files(data_dir_nacelle_9)
printar(audio_files_7, data_dir_nacelle_9)
result_7=resultado(audio_files_7,data_dir_nacelle_9)

#------------------------------------------------------------------------------------------

dt1 =  pd.DataFrame(result_1)
dt1.to_csv('PIPI_control_denoising.csv')

dt2 =  pd.DataFrame(result_2)
dt2.to_csv('PIPI_ground_1_denoising.csv')

dt3 =  pd.DataFrame(result_3)
dt3.to_csv('PIPI_nacelle_1_denoising.csv')

dt4 =  pd.DataFrame(result_4)
dt4.to_csv('PIPI_ground_5_denoising.csv')

dt5 =  pd.DataFrame(result_5)
dt5.to_csv('PIPI_nacelle_5_denoising.csv')

dt6 =  pd.DataFrame(result_6)
dt6.to_csv('PIPI_ground_9_denoising.csv')

dt7 =  pd.DataFrame(result_7)
dt7.to_csv('PIPI_nacelle_9_denoising.csv')
