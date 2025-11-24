python3.13.exe entrypoints/launch.py \
--model_path Wan-AI/Wan2.1-T2V-14B-Diffusers \
--world_size 4 \
--save_disk_path 'output\wan\sp' \
--data_parallel_degree 1 \
--pipefusion_parallel_degree 1 \
--ulysses_parallel_degree 4