1. In a process of feeding saved noisy latent state files (.pt) of Model A (seed=0) into different models at specific time steps (998, 900, 800, …, 200, 100, 0)
2. Difference between Training seed we’ve used for model trainings and Step seed that’s used during sampling; to see only the differences come from the models, we set the step seed at 777 for now
3. Successfully saved the final images generated from Model B (seed=42) and their final denoised image tensors (.pt) as well

Command used:

python sample_ddpm_inject_xt.py \
  --model_b /yunity/poil08/hugging_face/hf_cifar10_ddpm_safe_seed42 \
  --out_dir /yunity/poil08/cifar_branch/xt999_into_B_many_startt \
  --prefer_ema \
  --step_seed 777 \
  --inject \
    /yunity/poil08/cifar_branch/modelA_seed0_saved_xt/pt/x_t0999_batch00.pt:998 \
    /yunity/poil08/cifar_branch/modelA_seed0_saved_xt/pt/x_t0999_batch00.pt:900 \
    /yunity/poil08/cifar_branch/modelA_seed0_saved_xt/pt/x_t0999_batch00.pt:800 \
    /yunity/poil08/cifar_branch/modelA_seed0_saved_xt/pt/x_t0999_batch00.pt:700 \
    /yunity/poil08/cifar_branch/modelA_seed0_saved_xt/pt/x_t0999_batch00.pt:600 \
    /yunity/poil08/cifar_branch/modelA_seed0_saved_xt/pt/x_t0999_batch00.pt:500 \
    /yunity/poil08/cifar_branch/modelA_seed0_saved_xt/pt/x_t0999_batch00.pt:400 \
    /yunity/poil08/cifar_branch/modelA_seed0_saved_xt/pt/x_t0999_batch00.pt:300 \
    /yunity/poil08/cifar_branch/modelA_seed0_saved_xt/pt/x_t0999_batch00.pt:200 \
    /yunity/poil08/cifar_branch/modelA_seed0_saved_xt/pt/x_t0999_batch00.pt:100 \
    /yunity/poil08/cifar_branch/modelA_seed0_saved_xt/pt/x_t0999_batch00.pt:0 \
  --save_final_pt

## Timline of generated images (Model A (t=999) into Model B (t= 998, 900, 800, ..., 100, 0)
![timeline](timeline_grid.png)
