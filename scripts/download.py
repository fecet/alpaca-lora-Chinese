from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="decapoda-research/llama-30b-hf",
    cache_dir="data/hf",
    local_dir_use_symlinks=False,
    resume_download=True,
)
