import re
import os
from pytube import YouTube
import pandas as pd
from tqdm import tqdm


data = pd.read_csv('/home/amirbek.djanibekov/Documents/nlp703/project/nlp703-speech-processing/csv/usa_music_genre_wo_text.csv', encoding='UTF-8')
data['link'] = data['ondblclick'].apply(
    lambda x: (
            re.search(r'(?P<link>https.*)",',x) or
            re.search(r'(?P<link>.*)', x)
    ).group('link')
)
data['domain'] = data['ondblclick'].apply(
    lambda x: (
            re.search(r'www\.(?P<domain>.*)\.\w',x) or
            re.search(r'(?P<domain>.*)', x)
    ).group('domain')
)

data['domain'].value_counts(normalize=True)
youtube = data[data['domain'] == 'youtube']

def download_from_youtube(row):
    print('start')
    # time.sleep(5)
    try:
        yt = YouTube(row['link'])
        video = yt.streams.filter(only_audio=True).first()

        destination = f'music-full/{row["class"]}/'
        out_file = video.download(output_path=destination)

        os.rename(out_file, destination + row["id"] + '.mp3')
    except KeyError as e:
        print(str(e))
        with open('errors_while_downloading.txt', 'a') as file:
            file.write(row['id'] + ',' + row['link'])
    except Exception:
        with open('error.txt', 'a') as file:
            file.write(row['id'] + ', ' + row['link'])
            

tqdm.pandas()
# instantiate
youtube.progress_apply(download_from_youtube, axis=1)