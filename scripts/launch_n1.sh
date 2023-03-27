# export NCCL_IB_DISABLE=1;
# export NCCL_SOCKET_IFNAME="=ens42f1";
export NCCL_DEBUG=INFO;
torchrun \
    --rdzv_id=ID0 --rdzv_backend=c10d \
    --nproc_per_node=8 --nnodes=2 \
    --rdzv_endpoint=172.18.18.171:9999 \
    train_lora.ju.py
