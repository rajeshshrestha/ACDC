
BATCH_SIZE=9
TOTAL_IMAGES=9

ROOT_DIR=results/




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
    compression_quantization
)


data=(ffhq
imagenet)

for data in "${data[@]}"; do
    echo $sampler $data
    for task in "${tasks[@]}"; do
        for denoise_type in 'tweedie'; do
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
            sampler='edm_admm'\
            inverse_task.admm_config.denoise.final_step=$denoise_type
        done
    done
done
