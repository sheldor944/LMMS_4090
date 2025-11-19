PREFIX="selected_frames_dbfp_20_alpha_0.75_score_diff_power_law_power_2.0$(date +"%Y%m%d_%H")" && \
mkdir -p ./results/full_logs && \
nohup bash -c '
CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS="ignore" stdbuf -oL -eL python -u -m lmms_eval \
  --model llava_vid \
  --model_args pretrained=/home/hpc4090/miraj/AKS/AKS/llava_eval/LLaVA-NeXT-Video-7B-Qwen2,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=20,overwrite=False \
  --tasks custom_video_qa \
  --batch_size 1 \
  --device cuda:0 \
  --output_path ./results/full_logs/'"$PREFIX"'_results \
  --log_samples \
  --log_samples_suffix '"$PREFIX"' \
  --verbosity DEBUG \
  2>&1 | tee ./results/full_logs/'"$PREFIX"'.log
' > /dev/null 2>&1 &
disown
