set -ex
#for i in {1..9}
for i in 0; do
	echo $i
	python torchtune/_cli/tune.py run recipes/x_full_finetune_single_device.py \
		--config ./recipes/configs/qwen2/x_full_single_device.yaml
done
