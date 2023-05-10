# Per-interval emotion recognition of instrumental and vocal tracks of music
## Course Project
## By Akhmed Sakip and Amirbek Djanibekov

### Setting up the environment
First, create a separate conda environment with Python=3.8 to run this project:

`conda create --name nlp python=3.8`

`conda activate nlp`

Then, install PyTorch (together with *torchaudio*) with GPU support, following the guidelines [here](https://pytorch.org/get-started/locally/) for your particular CUDA version.

After installing PyTorch, install the other libraries required by the project by running the following in the **root** folder of the project:

`pip install -r requirements.txt`

Once the installation is done, you can proceed to passing your input music file through our pipeline. For this, create a folder in the project's root directory named *input/* and place an .mp3 or .wav music file there. You do not need to resample it to 16 KHz as this is done automatically. Then, run the following while in the project's **endtoend/** folder:

`cd endtoend`

`python inference.py`

Please note that the fine-tuned models for both instrumental and vocal emotion recognition will be automatically downloaded from Hugging Face. They can be accessed directly by following these links: [instrumental model](https://huggingface.co/akhmedsakip/wav2vec2-base-berkeley) and [vocal model](https://huggingface.co/akhmedsakip/wav2vec2-base-ravdess).

After the script successfully finishes its execution, the output files are stored at **endtoend/intermediate/**. To open the webpage and play the classified segments, just open the *endtoend/align.html* file in your browser.