from os import path
from pydub import AudioSegment

# files                                                                         
src = "/home/akhmed.sakip/Documents/NLP703/Project/nlp703-speech-processing/notebooks/music-full-mp3/item2.mp3"
dst = "/home/akhmed.sakip/Documents/NLP703/Project/nlp703-speech-processing/notebooks/music-full/item2.wav"

# convert wav to mp3                                                            
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")