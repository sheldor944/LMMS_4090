# PREFIX="dbfp_.85_3_new"

# mkdir -p ./results/full_logs

# nohup bash -c "
# CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS='ignore' stdbuf -oL -eL \
# python -u -m lmms_eval \
#   --model llava_vid \
#   --model_args pretrained=../LLaVA-NeXT-Video-7B-Qwen2,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=16,overwrite=False \
#   --tasks custom_video_qa \
#   --batch_size 1 \
#   --device cuda:0 \
#   --output_path ./results/full_logs/${PREFIX}_results \
#   --log_samples \
#   --log_samples_suffix ${PREFIX} \
#   --verbosity DEBUG \
#   2>&1 | tee ./results/full_logs/${PREFIX}.log
# " > /dev/null 2>&1 &

# disown



PREFIX="selected_frames_ada_dq_k32_beta0.5_delta3.0"
# PREFIX="1selected_dbfp_videomme_clip_k16_alpha0.85_sup10_score_diff_minscore0.7_TOD100$(date +"%Y%m%d_%H")"
mkdir -p ./results/full_logs

# nohup bash -c '
# CUDA_VISIBLE_DEVICES=0,1 PYTHONWARNINGS="ignore" stdbuf -oL -eL accelerate launch --num_processes 2 --main_process_port 29500 -m lmms_eval \
#   --model llava_vid \
#   --model_args pretrained=../LLaVA-NeXT-Video-7B-Qwen2,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=16,overwrite=False,device_map=auto \
#   --tasks longvideobench_custom \
#   --batch_size 1 \
#   --output_path ./results/full_logs/'"$PREFIX"'_results \
#   --log_samples \
#   --log_samples_suffix '"$PREFIX"' \
#   --verbosity DEBUG \
#   --limit 4\
#   2>&1 | tee ./results/full_logs/'"$PREFIX"'.log
# ' > /dev/null 2>&1 &
# disown





# ==========================
# ACcelerate
# ==========================
# nohup bash -c '
# CUDA_VISIBLE_DEVICES=0,1 PYTHONWARNINGS="ignore" stdbuf -oL -eL accelerate launch --num_processes 2 --main_process_port 29500 -m lmms_eval \
#   --model llava_vid \
#   --model_args pretrained=../LLaVA-NeXT-Video-7B-Qwen2,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=16,overwrite=False,device_map=auto \
#   --tasks longvideobench_custom \
#   --batch_size 1 \
#   --output_path ./results/full_logs/'"$PREFIX"'_results \
#   --log_samples \
#   --log_samples_suffix '"$PREFIX"' \
#   --verbosity DEBUG \
#   2>&1 | tee ./results/full_logs/'"$PREFIX"'.log
# ' > /dev/null 2>&1 &
# disown

# ====================================
# The one below works fine shit
# ==================================

# nohup bash -c '
# CUDA_VISIBLE_DEVICES=0,1 PYTHONWARNINGS="ignore" stdbuf -oL -eL accelerate launch --num_processes 2 --main_process_port 29500 -m lmms_eval \
#   --model llava_vid \
#   --model_args pretrained=../LLaVA-NeXT-Video-7B-Qwen2,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=24,overwrite=False,device_map=auto \
#   --tasks longvideobench_custom \
#   --batch_size 1 \
#   --output_path ./results/full_logs/'"$PREFIX"'_results \
#   --log_samples \
#   --log_samples_suffix '"$PREFIX"' \
#   --verbosity DEBUG \
#   2>&1 | tee ./results/full_logs/'"$PREFIX"'.log
# ' > /dev/null 2>&1 &
# disown


nohup bash -c '
CUDA_VISIBLE_DEVICES=1,0 PYTHONWARNINGS="ignore" stdbuf -oL -eL python -m lmms_eval \
  --model llava_vid \
  --model_args pretrained=../LLaVA-NeXT-Video-7B-Qwen2,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=32,overwrite=False,device_map=auto \
  --tasks longvideobench_custom \
  --batch_size 1 \
  --output_path ./results/full_logs/'"$PREFIX"'_results \
  --log_samples \
  --log_samples_suffix '"$PREFIX"' \
  --verbosity DEBUG \
  2>&1 | tee ./results/full_logs/'"$PREFIX"'.log
' > /dev/null 2>&1 &
disown








