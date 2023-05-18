# To run this script, type the following line in the powershell inside the main folder 
# powershell -ExecutionPolicy Bypass -File automatic_runner.ps1

#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/cnnRotatedEmnist.py --num_epochs 20 --lr 0.01 --bs 128

#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --niid --num_epochs 1 --clients_per_round 10

#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --dataset_selection rotated

C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/cnnRotatedEmnist.py --dataset_selection L1O --leftout 2 --num_epochs 20 --lr 0.01 --bs 128
C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --dataset_selection L1O --leftout 3
C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/cnnRotatedEmnist.py --dataset_selection L1O --leftout 3 --num_epochs 20 --lr 0.01 --bs 128
C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --dataset_selection L1O --leftout 4
C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/cnnRotatedEmnist.py --dataset_selection L1O --leftout 4 --num_epochs 20 --lr 0.01 --bs 128
C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --dataset_selection L1O --leftout 5
C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/cnnRotatedEmnist.py --dataset_selection L1O --leftout 5 --num_epochs 20 --lr 0.01 --bs 128