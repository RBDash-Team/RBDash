#!/bin/bash
#
CUDA_VISIBLE_DEVICES='0,1|2,3|4,5|6,7'
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS='|' read -ra GPULIST <<<"$gpu_list"
CHUNKS=${#GPULIST[@]}

CKPT="RBDash-v1.2-72b"

for IDX in $(seq 0 $((CHUNKS - 1))); do
	CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m rbdash.eval.model_vqa_loader \
		--model-path models/$CKPT \
		--question-file rbdash-Eval/MME/llava_mme.jsonl \
		--image-folder rbdash-Eval/MME/MME_Benchmark_release_version \
		--answers-file rbdash-Eval/MME/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
		--temperature 0 \
		--num-chunks $CHUNKS \
		--chunk-idx $IDX \
		--conv-mode qwen &
done

wait

cat rbdash-Eval/MME/answers/$CKPT/*.jsonl >rbdash-Eval/MME/answers/$CKPT.jsonl

cd rbdash-Eval/MME

python convert_answer_to_mme.py --experiment $CKPT

cd eval_tool

python calculation.py --results_dir answers/$CKPT
