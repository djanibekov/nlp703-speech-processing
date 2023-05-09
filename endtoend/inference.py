from transformers import Wav2Vec2FeatureExtractor, AutoConfig
from datasets import load_dataset, load_metric
from sklearn.metrics import classification_report
import numpy as np
from torch import nn
import torch
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification
import string
import torchaudio
import librosa
import numpy as np
import pandas as pd
import librosa as lr
import soundfile as sf
import os
import re
import subprocess
import shutil
from IPython.display import Audio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import sys

input_dir = "../input/"
intermediate_dir = "intermediate/"
separated_dir = os.path.join(intermediate_dir, "separated") + "/"
segmented_dir = os.path.join(intermediate_dir, "segmented") + "/"
relative_segmented_dir = "intermediate/segmented/"
csv_dir = os.path.join(intermediate_dir, "csv") + "/"
relative_csv_dir = "intermediate/csv/"

# Deleting possibly created direcotires and creating the necessary ones
if os.path.exists(intermediate_dir):
    shutil.rmtree(intermediate_dir)
os.mkdir(intermediate_dir)
os.mkdir(separated_dir)
os.mkdir(segmented_dir)
os.mkdir(csv_dir)

# Get all .wav and .mp3 files in input_dir
audio_files = [f for f in os.listdir(input_dir) if f.endswith(".wav") or f.endswith(".mp3")]

# Check that there is at most one .wav or .mp3 file in input_dir
if len(audio_files) > 1:
    print("Error: There should be at most one .wav or .mp3 file in input_dir.")
    sys.exit(1)
elif len(audio_files) == 0:
    print("Error: There are no .wav or .mp3 files in input_dir.")
    sys.exit(1)

separated_dir_abs = os.path.abspath(separated_dir)

wrk_dir = os.getcwd()
os.chdir("../vocal-remover")

subprocess.run(["python", "inference_dir.py", 
                "--input_dir", input_dir, 
                "--output_dir", separated_dir_abs, 
                "--gpu", "0"])

os.chdir(wrk_dir)

files = os.listdir(separated_dir)

target_sr = 16000
instruments, vocals = None, None

for file in files:
    y, sr = lr.load(os.path.join(separated_dir, file))
    y_16k = lr.resample(y, orig_sr=sr, target_sr=target_sr)
    file_without_extension = re.sub(r"\.\w+$", "", file)
    if "Instruments" in file:
        instruments = y_16k
        sf.write(os.path.join(separated_dir, file_without_extension) + '.wav', y_16k, target_sr)
    elif "Vocals" in file:
        vocals = y_16k
        sf.write(os.path.join(separated_dir, file_without_extension) + '.wav', y_16k, target_sr)
        
def split_into_segments(y, sr, segment_duration=5):
    frames_per_segment = segment_duration * sr
    
    num_segments = int(len(y) / frames_per_segment)
    
    segments = []

    for i in range(num_segments):
        start = i * frames_per_segment
        end = (i + 1) * frames_per_segment
        segment = y[start:end]
        segments.append(segment)

    return segments


instruments_segments = split_into_segments(instruments, target_sr)

vocals_segments = split_into_segments(vocals, target_sr)

def save_segments(path, segments, sr):
    df = pd.DataFrame(columns=['path', 'segment_number'])
    for i in range(len(segments)):
        full_path = path + '_' + str(i) + '.wav'
        sf.write(full_path, segments[i], sr)
        df = pd.concat([df, pd.DataFrame({'path': full_path, 'segment_number': i}, index=[i])])
    return df

df_instruments = save_segments(relative_segmented_dir + "i", instruments_segments, target_sr)

df_instruments.to_csv(os.path.join(csv_dir, "instruments_segments.csv"), index=False)

df_vocals = save_segments(relative_segmented_dir + "v", vocals_segments, target_sr)

df_vocals.to_csv(os.path.join(csv_dir, "vocals_segments.csv"), index=False)

def heard_segments(segments, sr, threshold):
    heard = []
    for i, segment in enumerate(segments):
        rms = np.sqrt(np.sum(segment**2))
        if rms > threshold:
            heard.append(i)
    return heard

heard_vocals = heard_segments(vocals_segments, target_sr, 10)

heard_vocals

df_vocals_heard = df_vocals[df_vocals['segment_number'].isin(heard_vocals)]
df_vocals_heard

df_vocals_heard.to_csv(os.path.join(csv_dir, "vocals_heard_segments.csv"), index=False)

df_instruments = pd.read_csv(os.path.join(csv_dir, 'instruments_segments.csv'))
df_vocals = pd.read_csv(os.path.join(csv_dir, 'vocals_segments.csv'))
df_vocals_heard = pd.read_csv(os.path.join(csv_dir, 'vocals_heard_segments.csv'))

# Inference

max_audio_len = 10
model_checkpoint_trained = "akhmedsakip/wav2vec2-base-berkeley"
test_dataset = load_dataset("csv", data_files={
    "test": os.path.join(csv_dir, "instruments_segments.csv")
    })
label2id, id2label = dict(), dict()
labels = list(string.ascii_uppercase)[:13]

id2label = {'0': 'fearful', '1': 'amusing', '2': 'sad', 
            '3': 'beautiful', '4': 'anxious', '5': 'erotic', 
            '6': 'dreamy', '7': 'calm', '8': 'joyful', 
            '9': 'energizing', '10': 'triumphant', '11': 'annoying', '12': 'indignant'}
label2id = {v:k for k, v in id2label.items()}

num_labels = len(id2label)

id2label

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = AutoConfig.from_pretrained(model_checkpoint_trained)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint_trained)
sampling_rate = feature_extractor.sampling_rate
model = model = AutoModelForAudioClassification.from_pretrained(
    model_checkpoint_trained,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
).to(device)

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array.squeeze().numpy()
    # speech_array = librosa.resample(np.asarray(speech_array), orig_sr=sampling_rate, target_sr=feature_extractor.sampling_rate)

    batch["audio"] = speech_array
    return batch

def preprocess_function(examples):
    audio_arrays = [x for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate*int(max_audio_len)),
        # truncation=True, # Uncomment it, If you want to truncate longer audios to max_length
        padding=True, # Uncomment it, if you want to pad shorter audio to max_length
        return_tensors="pt",
    )
    return inputs

def predict(batch):
    features = preprocess_function(batch)

    input_values = features.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits 

    m = nn.Softmax(dim=1)
    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    batch["predicted"] = pred_ids
    batch['probas'] = m(logits).detach().cpu().numpy()
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)

result = test_dataset.map(predict, batched=True, batch_size=8)['test']

label_names = [config.id2label[i] for i in range(config.num_labels)]

# y_true = [int(config.label2id[name]) for name in result["label"]]
y_pred = result["predicted"]

instruments_predictions = list(map(lambda x: id2label[str(x)], y_pred))

instruments_predictions

instruments_predictions_with_segment = list(zip(result['segment_number'], instruments_predictions))

max_audio_len = 10
model_checkpoint_trained = "akhmedsakip/wav2vec2-base-ravdess"
test_dataset = load_dataset("csv", data_files={
    "test": os.path.join(csv_dir, "vocals_heard_segments.csv")
    })

id2label = {
    "0": "fearful",
    "1": "neutral",
    "2": "calm",
    "3": "happy",
    "4": "sad",
    "5": "angry",
    }
label2id = {v:k for k, v in id2label.items()}

num_labels = len(id2label)

test_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = AutoConfig.from_pretrained(model_checkpoint_trained)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint_trained)
sampling_rate = feature_extractor.sampling_rate
model = model = AutoModelForAudioClassification.from_pretrained(
    model_checkpoint_trained,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
).to(device)

test_dataset = test_dataset.map(speech_file_to_array_fn)

result = test_dataset.map(predict, batched=True, batch_size=8)['test']

label_names = [config.id2label[i] for i in range(config.num_labels)]

# y_true = [int(config.label2id[name]) for name in result["label"]]
y_pred = result["predicted"]

vocals_predictions = list(map(lambda x: id2label[str(x)], y_pred))

vocals_predictions_with_segments = list(zip(result['segment_number'], vocals_predictions))

df = pd.DataFrame(columns=['segment_number', 'instrument_emotion', 'vocal_emotion'])
segment_number = range(len(instruments_predictions))
df['segment_number'] = segment_number
df.set_index('segment_number', inplace=True)

for index, emotion in vocals_predictions_with_segments:
    df['vocal_emotion'].iloc[index] = emotion
    
for index, emotion in instruments_predictions_with_segment:
    df['instrument_emotion'].iloc[index] = emotion
    
df['vocal_emotion'].fillna('-', inplace=True)
df.reset_index(inplace=True)
resulting_pair_emotions = os.path.join(csv_dir, 'instr_vocal_emotions.csv')
df.to_csv(resulting_pair_emotions, index=False)

print(f'Results are saved to {resulting_pair_emotions}')