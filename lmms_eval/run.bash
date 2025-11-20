# PREFIX="selected_dbfp_longvideobench_blip_k16_alpha0.75_sup3_score_diff$(date +"%Y%m%d_%H")" && \
# mkdir -p ./results/full_logs && \
# nohup bash -c '
# CUDA_VISIBLE_DEVICES=0,1 PYTHONWARNINGS="ignore" stdbuf -oL -eL python -u -m lmms_eval \
#   --model llava_vid \
#   --model_args pretrained=../LLaVA-NeXT-Video-7B-Qwen2,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=16,overwrite=False \
#   --tasks longvideobench_custom \
#   --batch_size 1 \
#   --device cuda:0 \
#   --output_path ./results/full_logs/'"$PREFIX"'_results \
#   --log_samples \
#   --log_samples_suffix '"$PREFIX"' \
#   --verbosity DEBUG \
  
#   2>&1 | tee ./results/full_logs/'"$PREFIX"'.log
# ' > /dev/null 2>&1 &
# disown



PREFIX="Full_TEST_newUtils$(date +"%Y%m%d_%H")"
mkdir -p ./results/full_logs

nohup bash -c '
CUDA_VISIBLE_DEVICES=0,1 PYTHONWARNINGS="ignore" stdbuf -oL -eL accelerate launch --num_processes 2 --main_process_port 29500 -m lmms_eval \
  --model llava_vid \
  --model_args pretrained=../LLaVA-NeXT-Video-7B-Qwen2,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=16,overwrite=False,device_map=auto \
  --tasks longvideobench_custom \
  --batch_size 1 \
  --output_path ./results/full_logs/'"$PREFIX"'_results \
  --log_samples \
  --log_samples_suffix '"$PREFIX"' \
  --verbosity DEBUG \
  --limit 4\
  2>&1 | tee ./results/full_logs/'"$PREFIX"'.log
' > /dev/null 2>&1 &
disown



# PREFIX="selected_dbfp_videomme_blip_k32_alpha0.75_sup3.0_score_diff$(date +"%Y%m%d_%H")" && \
# mkdir -p ./results/full_logs && \
# nohup bash -c '
# CUDA_VISIBLE_DEVICES=0,1 PYTHONWARNINGS="ignore" stdbuf -oL -eL \
# accelerate launch --num_processes 2 --num_machines 1 --multi_gpu \
#     --mixed_precision bf16 --dynamo_backend no \
#     -m lmms_eval \
#         --model llava_vid \
#         --model_args pretrained=../LLaVA-NeXT-Video-7B-Qwen2,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=64,overwrite=False \
#         --tasks custom_video_qa \
#         --batch_size 1 \
#         --device cuda \
#         --output_path ./results/full_logs/'"$PREFIX"'_results \
#         --log_samples \
#         --log_samples_suffix '"$PREFIX"' \
#         --verbosity DEBUG \
#     2>&1 | tee ./results/full_logs/'"$PREFIX"'.log
# ' > /dev/null 2>&1 &
# disown
