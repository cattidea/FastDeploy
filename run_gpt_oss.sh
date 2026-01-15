rm -rf log/*

# /root/paddlejob/tmpspace/executable/nsight-systems/2024.3.1/bin/nsys
# MODEL=/root/paddlejob/tmpspace/MODELS/ERNIE-4.5-0.3B-Paddle
MODEL=/root/paddlejob/tmpspace/MODELS/ERNIE-4.5-21B-A3B-Paddle
MODEL=/root/paddlejob/tmpspace/MODELS/ERNIE-4.5-0.3B-Paddle

PORT=39899
let MPPPPP=PORT+1
let EWQPPP=PORT+2


# export FLAGS_use_system_allocator=1
# export FLAGS_check_cuda_error=1
# export FD_DEBUG=1
# export FLAGS_cuda_graph_blacklist="custom_op.static_op_append_attention_with_output_,custom_op.static_op_append_attention"
export PYTHONPATH=/root/paddlejob/tmpspace/huangzihao/FastDeploy
# export SOT_LOG_LEVEL=3
export FLAGS_print_ir=1
export CUDA_VISIBLE_DEVICES=2,3,4,5
# export GLOG_v=6
# export FLAGS_call_stack_level=2

python -m fastdeploy.entrypoints.openai.api_server \
       --model $MODEL \
       --port $PORT \
       --metrics-port $MPPPPP \
       --engine-worker-queue-port $EWQPPP \
       --tensor-parallel-size 4 \
       --max-model-len 32768 \
       --max-num-seqs 128 \
       --graph-optimization-config '{"graph_opt_level": 1, "use_cudagraph":true, "full_cuda_graph": false}' \
       --quantization wint4 \
       # --disable-custom-all-reduce

# curl -X POST "http://0.0.0.0:19823/v1/chat/completions" \
# -H "Content-Type: application/json" \
# -d '{
#   "messages": [
#     {"role": "user", "content": "浣犳槸璋�?????"}
#   ]
# }'
