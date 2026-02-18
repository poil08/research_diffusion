1) Trying to figure out why the model A has generated the same images without the given values to its noise seed
  - Possibility 1: deterministic flags somewhere (if CUDA deterministic settings were enabled)
  - Possibility 2: scheduler randomness dominates (if schedular variance is small or deterministic)

2) Checked the json file of scheduler; didn't find anything that could possibly interrupt the model's generalizability randomness

 
