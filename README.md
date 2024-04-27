# Midi-Text-Generator
Code for Yeerchiu/MIDI-TEXT  

Adding text input for MIDI transformer model

MIDI-Text pair data generation utilize of [MU-LLaMA](https://github.com/shansongliu/MU-LLaMA)

if you want to run app.py, at least one ckpt file is needed above(Yeerchiu/MIDI-TEXT).

* At present, the code is run in an environment with GPU. There is no proof yet to see if it can run with only CPU.
* The average GPU memory usage during the runtime of this model is about 5-6GB
# TODO
* Due to lack of computing resources and time, only 20,000+ MIDI files collected from Free MIDI Web were used on the MIDI-Text pair training. 
* There may be some address errors that have not been corrected.
* The text can influence the model generation but cannot always get the desirable results, optimization is still needed.
  


# Prepare the environment
- Follow the official guidance for configuring the GPU graphics drivers, CUDA, and cuDNN, etc.
- Create a new environment
<pre>
conda create midi -n python=3.10
conda activate midi
</pre>
- Install Pytorch following the <a>https://pytorch.org/</a>
- Install packages
<pre>
    conda install --yes --file requirements.txt
    or
    pip install -r requirements.txt
</pre>
# Reference
[Original Model](https://github.com/SkyTNT/midi-model)  
[Training Dataset](https://github.com/asigalov61/Los-Angeles-MIDI-Dataset)
# Device 
MIDI-Text pair data generated on V100  

Training on RTX 4090  
App Tested in RTX 4060 (Laptop)




