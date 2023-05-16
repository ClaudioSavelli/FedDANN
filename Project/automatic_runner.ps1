# To run this script, type the following line in the powershell inside the main folder 
# powershell -ExecutionPolicy Bypass -File automatic_runner.ps1

C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/cnnEmnist.py --lr 0.1 --num_epochs 20 --bs 128
C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/cnnEmnist.py --lr 0.01 --num_epochs 20 --bs 128
C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/cnnEmnist.py --lr 0.001 --num_epochs 20 --bs 128

C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/cnnEmnist.py --lr 0.1 --num_epochs 20 --change_lr_interval 3 --bs 128
C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/cnnEmnist.py --lr 0.1 --num_epochs 20 --change_lr_interval 5 --bs 128