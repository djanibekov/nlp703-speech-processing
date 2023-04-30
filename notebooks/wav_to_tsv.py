import os
import pandas as pd
import librosa as lr
from tqdm import tqdm

def process_file(file):
    audio, sr = lr.load(os.path.join(path, file))
    audio_dict = dict(zip(range(len(audio)), audio))
    audio_dict['filename'] = file
    return audio_dict

path = 'music-segments/'
files = [f for f in os.listdir(path) if f.endswith('.wav')]

data_list = []
for file in tqdm(files):
    audio_data = process_file(file)
    data_list.append(audio_data)
    
dataframes = pd.DataFrame(data_list)

dataframes.to_csv('combined_music_data.tsv', sep='\t', index=False)