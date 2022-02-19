import os,sys,GPUtil
import time

def mem_report(GPUs):
  for i, gpu in enumerate(GPUs):
    return gpu.memoryFree

while True:
	GPUs = GPUtil.getGPUs()
	time.sleep(4)
	space = mem_report(GPUs)
	print(space)
	if space >= 11000:
		os.system('python3 main_supcon.py --dataset path --data_folder custom_dataset_split/ --mean 0.41044191,0.45704237,0.46365224 --std 4.37454361,4.06389989,4.11659655 --batch_size 512 --epochs 3200 --learning_rate 1.4 --model resnet34  --exp')
