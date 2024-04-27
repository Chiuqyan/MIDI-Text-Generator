# MidiLLMBasedGenerator
Code for Yeerchiu/midi-llama-test1

* At present, the code is run in an environment with GPU. There is no proof yet to see if it can run with only CPU.
* The average GPU memory usage during the runtime of this model is about 5-6GB
* There may be some address errors that have not been corrected.
  
if you want to run app.py, at least one ckpt file is needed above(Yeerchiu/midi-llama-test1).

# Prepare the environment
- According to the official configuration GPU graphics driver, etc.
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




