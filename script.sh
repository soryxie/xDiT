SP=4
PP=1

python entrypoints/launch.py \
  --model_path Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --world_size 4 \
  --save_disk_path 'output\wan\sp' \
  --data_parallel_degree 1 \
  --pipefusion_parallel_degree ${PP} \
  --ulysses_parallel_degree ${SP} \
  > "/home/jovyan/shared/yqiao/sparfusionstorage/baseline/xdit/$(date +%Y%m%d_%H%M%S)_SP_${SP}_PP_${PP}.log" 2>&1