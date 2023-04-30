import numpy as np
import pandas as pd
import librosa as lr
import soundfile as sf
import re
import os
from IPython.display import Audio

def split_audio_into_segments(audio_file, segment_duration=10):
    # Load the audio file
    y, sr = lr.load(audio_file)
    
    # Calculate the number of frames per segment
    frames_per_segment = segment_duration * sr
    
    # Calculate the number of segments
    num_segments = int(len(y) / frames_per_segment)
    
    # Initialize a list to store the segments
    segments = []

    # Extract and store the segments
    for i in range(num_segments):
        start = i * frames_per_segment
        end = (i + 1) * frames_per_segment
        segment = y[start:end]
        segments.append(segment)

    return segments

def save_segments(path, segments, sr):
    path_no_ext = re.sub('\.[^.]*$', '', path)
    for i in range(len(segments)):
        sf.write(path_no_ext + '_' + str(i) + '.wav', segments[i], sr)

def list_files_smaller_than_5mb(directory):
    file_list = []

    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)

        # Check if it's a file and not a directory
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)

            # 5MB in bytes
            max_size = 5 * 1024 * 1024

            if file_size < max_size:
                file_list.append(file)

    return file_list

if __name__ == "__main__":
    dir = '/home/akhmed.sakip/Documents/NLP703/Project/nlp703-speech-processing/notebooks/music-full-mp3/'
    files = list_files_smaller_than_5mb(dir)
    # files = os.listdir(dir)
    # print(len(files))
    
    sr = 22050
    i = 0
    
    for file in files:
        path = dir + file
        print(path)
        segments = split_audio_into_segments(path, segment_duration=10)
        
        if len(segments) < 1:
            continue
        
        save_segments(path, segments, sr)
        
        print(f'Saved file {file}')