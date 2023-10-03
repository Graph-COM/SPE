# python runner.py --config_dirpath ../configs/zinc --config_name SPE_gine_gin_mlp_pe37.yaml --seed 0
for j in 0 1 2
do 
for t in 0 1 2 3
do
	python runner.py --config_dirpath ../configs/count --config_name SPE_gine_gin_mlp.yaml --seed $j --target $t --gpu_id 3
	python runner.py --config_dirpath ../configs/count --config_name signNet_gine_mask.yaml --seed $j --target $t --gpu_id 3
	python runner.py --config_dirpath ../configs/count --config_name gine_peg.yaml --seed $j --target $t --gpu_id 3
	python runner.py --config_dirpath ../configs/count --config_name gine_nope.yaml --seed $j --target $t --gpu_id 3
done
done

