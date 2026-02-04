1. In a process of feeding saved noisy latent state files (.pt) of Model A (seed=0) into different models at specific time steps (998, 900, 800, …, 200, 100, 0)
2. Difference between Training seed we’ve used for model trainings and Step seed that’s used during sampling; to see only the differences come from the models, we set the step seed at 777 for now
3. Successfully saved the final images generated from Model B (seed=42) and their final denoised image tensors (.pt) as well

## Timline of generated images (Model A (t=999) into Model B (t= 998, 900, 800, ..., 100, 0))
![timeline](timeline_grid.png)
