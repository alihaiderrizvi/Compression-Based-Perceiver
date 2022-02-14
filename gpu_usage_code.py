import os,sys,humanize,psutil,GPUtil
import time

# Define function
def mem_report(GPUs):
  for i, gpu in enumerate(GPUs):
    return gpu.memoryFree


while True:
	GPUs = GPUtil.getGPUs()
	time.sleep(4)
	space = mem_report(GPUs)
	print(space)
	if space > 1000:
		os.system('python3 main_supcon.py --batch_size 1024 --learning_rate 100 --temp 0.07 --cosine --epochs 3000 --model resnet18 --dataset cifar10 --method SupCon')
