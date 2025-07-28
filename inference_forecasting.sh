prediction_length=61 # 31

exp_dir='./exp'
config='NeuralOM' 
run_num='20250309-195251'
finetune_dir='6_steps_finetune'

ics_type='default'

CUDA_VISIBLE_DEVICES=2 python inference_forecasting.py --exp_dir=${exp_dir} --config=${config} --run_num=${run_num} --finetune_dir=$finetune_dir --prediction_length=${prediction_length} --ics_type=${ics_type}



