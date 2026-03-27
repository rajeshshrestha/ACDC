
BATCH_SIZE=9
TOTAL_IMAGES=9

ROOT_DIR=results/


Y_SIGMA=0.05
ADMM_MAX_ITER=110
ANNEALING_STEPS=100

data=ffhq
echo $sampler $data
datetime_str=`date`

mkdir -p $ROOT_DIR/$data

tasks=( down_sampling
    gaussian_blur
    hdr
    inpainting_rand
    inpainting
    motion_blur
    phase_retrieval
    nonlinear_blur
    )

for task in "${tasks[@]}"; do
    for denoise_type in 'tweedie' 'ode'; do
            mkdir -p $ROOT_DIR/$data

            echo ====================================================================================
            echo Running ADMM for $task in $data with $denoise_type denoiser with sigma=$Y_SIGMA
            echo ====================================================================================
            python recover_inverse.py --config-name default_$data.yaml \
            sampler=edm_admm \
            inverse_task=$task \
            save_dir=$ROOT_DIR/$data/$task/ADMM-$denoise_type \
            batch_size=$BATCH_SIZE \
            total_images=$TOTAL_IMAGES \
            inverse_task.operator.sigma=$Y_SIGMA \
            sampler='edm_admm'\
            sampler.annealing_scheduler_config.num_steps=$ANNEALING_STEPS \
            inverse_task.admm_config.denoise.final_step=$denoise_type \
            inverse_task.admm_config.max_iter=$ADMM_MAX_ITER
        
    done
done
