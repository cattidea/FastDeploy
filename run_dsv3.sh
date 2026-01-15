rm -rf log/*

# export FLAGS_print_ir=1

# /root/paddlejob/tmpspace/executable/nsight-systems/2024.3.1/bin/nsys

MODEL=/root/paddlejob/tmpspace/bos_path/huggingface/deepseek-ai/DeepSeek-V3-0324-bf16
MODEL=/root/paddlejob/tmpspace/MODELS/DeepSeek-V3-0324-bf16-5layer
MODEL=/root/paddlejob/tmpspace/MODELS/10.95.246.25:8023/DeepSeek-V3-0324
MODEL=/root/paddlejob/tmpspace/MODELS/xiege_DeepSeek-V3-0324-bf16-5layer
# MODEL=/root/paddlejob/tmpspace/MODELS/DeepSeek-V3-0324-bf16
let MPPPPP=PORT+1
let EWQPPP=PORT+2

export FD_ATTENTION_BACKEND="MLA_ATTN"
export FLAGS_mla_use_tensorcore="1"
export FLAGS_flash_attn_version="3"
export FD_SAMPLING_CLASS=rejection
export FD_USE_MACHETE=1


# export SOT_LOG_LEVEL=3
# export SOT_EVENT_LEVEL=3
# export FLAGS_call_stack_level=2
# export GLOG_v=5
# export FLAGS_print_ir=1

export FLAGS_cuda_graph_blacklist="pd_op.if"
export PYTHONPATH=/root/paddlejob/tmpspace/huangzihao/FastDeploy:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
PORT=9905

python -m fastdeploy.entrypoints.openai.api_server \
  --model $MODEL \
  --metrics-port 39717 \
  --port $PORT \
  --engine-worker-queue-port 39719 \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --max-num-seqs 256 \
  --load-choices default_v1 \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --quantization wint4 \
  --graph-optimization-config '{"graph_opt_level": 1, "use_cudagraph": true, "full_cuda_graph": false, "use_unique_memory_pool": true}' \
