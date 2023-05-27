# To run this script, type the following line in the powershell inside the main folder 
# powershell -ExecutionPolicy Bypass -File automatic_runner.ps1

#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/cnnRotatedEmnist.py --num_epochs 20 --lr 0.01 --bs 128

#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --niid --num_epochs 1 --clients_per_round 10

#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --dataset_selection rotated

#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/cnnRotatedEmnist.py --dataset_selection L1O --leftout 2 --num_epochs 20 --lr 0.01 --bs 128
#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --dataset_selection L1O --leftout 3

#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --client_selection biased1 --clients_per_round 10
#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --client_selection biased1 --clients_per_round 10 --niid
#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --client_selection biased2 --clients_per_round 10
#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --client_selection biased2 --clients_per_round 10 --niid

#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --clients_per_round 10 --num_epochs 10 --niid   
#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --clients_per_round 20 --num_epochs 1 --niid 
#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --clients_per_round 20 --num_epochs 5 --niid
#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --clients_per_round 20 --num_epochs 10--niid        

#quelli da fare dopo appena sappiamo come fare con l1o

#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/cnnRotatedEmnist.py --dataset_selection rotated --num_epochs 20 --lr 0.01 --bs 128
#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/cnnRotatedEmnist.py --dataset_selection L1O --leftout 0 --num_epochs 20 --lr 0.01 --bs 128
#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/cnnRotatedEmnist.py --dataset_selection L1O --leftout 1 --num_epochs 20 --lr 0.01 --bs 128
#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/cnnRotatedEmnist.py --dataset_selection L1O --leftout 2 --num_epochs 20 --lr 0.01 --bs 128
#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/cnnRotatedEmnist.py --dataset_selection L1O --leftout 3 --num_epochs 20 --lr 0.01 --bs 128
#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/cnnRotatedEmnist.py --dataset_selection L1O --leftout 4 --num_epochs 20 --lr 0.01 --bs 128
#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/cnnRotatedEmnist.py --dataset_selection L1O --leftout 5 --num_epochs 20 --lr 0.01 --bs 128

C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --dataset_selection rotated
C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --dataset_selection L1O --leftout 0
C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --dataset_selection L1O --leftout 1
C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --dataset_selection L1O --leftout 2
C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --dataset_selection L1O --leftout 3
C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --dataset_selection L1O --leftout 4
C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --dataset_selection L1O --leftout 5

#Capire perchè non valuta il l1O questo metodo 
#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --dataset_selection rotated --prob --l2r 1e-2 --cmi 5e-4 --z_dim 1024 --model fedsr
#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --dataset_selection L1O --leftout 0 --prob --l2r 1e-2 --cmi 5e-4 --z_dim 1024 --model fedsr
#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --dataset_selection L1O --leftout 1 --prob --l2r 1e-2 --cmi 5e-4 --z_dim 1024 --model fedsr
#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --dataset_selection L1O --leftout 2 --prob --l2r 1e-2 --cmi 5e-4 --z_dim 1024 --model fedsr
#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --dataset_selection L1O --leftout 3 --prob --l2r 1e-2 --cmi 5e-4 --z_dim 1024 --model fedsr
#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --dataset_selection L1O --leftout 4 --prob --l2r 1e-2 --cmi 5e-4 --z_dim 1024 --model fedsr
#C:/ProgramData/miniconda3/python.exe d:/GitHub/Machine-Learning-and-Deep-Learning-Project/Project/main.py --dataset_selection L1O --leftout 5 --prob --l2r 1e-2 --cmi 5e-4 --z_dim 1024 --model fedsr