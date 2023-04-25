# vocal-remover-nlp703

## Usage
The following command separates the input into instrumental and vocal tracks. They are saved as `*_Instruments.wav` and `*_Vocals.wav`.

### Run on CPU
```
python inference_dir.py --input-dir path/to/an/input/dir --output-dir path/to/an/output/dir
```

### Run on GPU
```
python inference_dir.py --input-dir path/to/an/input/dir --output-dir path/to/an/output/dir --gpu 0
```

### Advanced options
`--tta` option performs Test-Time-Augmentation to improve the separation quality.
```
python inference_dir.py ---input-dir path/to/an/input/dir --output-dir path/to/an/output/dir --tta --gpu 0
```

`--postprocess` option masks instrumental part based on the vocals volume to improve the separation quality.  
**Experimental Warning**: If you get any problems with this option, please disable it.
```
python inference_dir.py --input-dir path/to/an/input/dir --output-dir path/to/an/output/dir --postprocess --gpu 0
```

## References
- [1] Jansson et al., "Singing Voice Separation with Deep U-Net Convolutional Networks", https://ejhumphrey.com/assets/pdf/jansson2017singing.pdf
- [2] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf
- [3] Takahashi et al., "MMDENSELSTM: AN EFFICIENT COMBINATION OF CONVOLUTIONAL AND RECURRENT NEURAL NETWORKS FOR AUDIO SOURCE SEPARATION", https://arxiv.org/pdf/1805.02410.pdf
- [4] Liutkus et al., "The 2016 Signal Separation Evaluation Campaign", Latent Variable Analysis and Signal Separation - 12th International Conference
