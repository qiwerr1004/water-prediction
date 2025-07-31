#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import numpy as np
from tqdm.auto import tqdm
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[5]:

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return np.random.normal(size = size, scale = xavier_stddev)

# Hint Vector Generation
def sample_M(m, n, p):
    A = np.random.uniform(0., 1., size = [m, n])
    B = A > p
    C = 1.*B
    return C

#%% 3. Other functions
# Random sample generator for Z
def sample_Z(m, n):
    return np.random.uniform(0., 0.01, size = [m, n])        

# Mini-batch generation
def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx

def gain(data_x, val_data, gain_parameters): 
    
    use_gpu = True

    #%% System Parameters
    # 1. Mini batch size
    mb_size = gain_parameters[0]#128
    # 2. Missing rate
    p_miss = gain_parameters[1]#0.2
    # 3. Hint rate
    p_hint = gain_parameters[2]#0.9
    # 4. Loss Hyperparameters
    alpha = gain_parameters[3]#10
    # 5. Train Rate
    train_rate = gain_parameters[4]#0.8
    iters = gain_parameters[5]#5000
    #%% Data

    # Data generation
    Data = data_x#np.loadtxt(dataset_file, delimiter=",",skiprows=1)
    size = len(Data)

    # Parameters
    No = len(Data) # 데이터 길이
    Dim = len(Data[0,:]) #데이터 차원 ex) 7

    # Hidden state dimensions
    H_Dim1 = Dim #히든 차원 수
    H_Dim2 = Dim #히든2 레이어 차원 수

    # normal 대체(0, 1)
    Data = scaler.fit_transform(Data)
    
    def renormal(data):
        min_value = scaler.data_min_
        max_value = scaler.data_max_
        data = data * (max_value - min_value) + min_value
        return data

#     # Normalization (0 to 1)
#     Min_Val = np.zeros(Dim)
#     Max_Val = np.zeros(Dim)

#     for i in range(Dim):
#         Min_Val[i] = np.min(Data[:,i])
#         Data[:,i] = Data[:,i] - np.min(Data[:,i])
#         Max_Val[i] = np.max(Data[:,i])
#         Data[:,i] = Data[:,i] / (np.max(Data[:,i]) + 1e-6)    

    #%% Missing introducing 
    p_miss_vec = p_miss * np.ones((Dim,1)) 
    Missing = np.zeros((No,Dim))

    for i in range(Dim):
        A = np.random.uniform(0., 1., size = [len(Data),])
        B = A > p_miss_vec[i]
        Missing[:,i] = 1.*B

    #%% Train Test Division    

    idx = np.random.permutation(No)

    Train_No = int(No * train_rate) #학습 데이터 수
    Test_No = No - Train_No # 테스트 데이터수

    # Train / Test Features
    trainX = Data[idx[:Train_No],:] #데이터수만큼 분할, 학습데이터
    testX = Data[idx[Train_No:],:] #데이터수만큼 분할, 테스트 데이터

    # Train / Test Missing Indicators
    trainM = Missing[idx[:Train_No],:] #앞에서 생성한 결측치 one hot 매트릭스 분할
    testM = Missing[idx[Train_No:],:] 
    
        #%% 1. Discriminator
    if use_gpu is True:
        D_W1 = torch.tensor(xavier_init([Dim*2, H_Dim1]),requires_grad=True, device="cuda")     # Data + Hint as inputs
        D_b1 = torch.tensor(np.zeros(shape = [H_Dim1]),requires_grad=True, device="cuda")
        D_W2 = torch.tensor(xavier_init([H_Dim1, H_Dim2]),requires_grad=True, device="cuda")
        D_b2 = torch.tensor(np.zeros(shape = [H_Dim2]),requires_grad=True, device="cuda")
        D_W3 = torch.tensor(xavier_init([H_Dim2, Dim]),requires_grad=True, device="cuda")
        D_b3 = torch.tensor(np.zeros(shape = [Dim]),requires_grad=True, device="cuda")       # Output is multi-variate
    else:
        D_W1 = torch.tensor(xavier_init([Dim*2, H_Dim1]),requires_grad=True)     # Data + Hint as inputs
        D_b1 = torch.tensor(np.zeros(shape = [H_Dim1]),requires_grad=True)

        D_W2 = torch.tensor(xavier_init([H_Dim1, H_Dim2]),requires_grad=True)
        D_b2 = torch.tensor(np.zeros(shape = [H_Dim2]),requires_grad=True)

        D_W3 = torch.tensor(xavier_init([H_Dim2, Dim]),requires_grad=True)
        D_b3 = torch.tensor(np.zeros(shape = [Dim]),requires_grad=True)       # Output is multi-variate

    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    #%% 2. Generator
    if use_gpu is True:
        G_W1 = torch.tensor(xavier_init([Dim*2, H_Dim1]),requires_grad=True, device="cuda")     # Data + Mask as inputs (Random Noises are in Missing Components)
        G_b1 = torch.tensor(np.zeros(shape = [H_Dim1]),requires_grad=True, device="cuda")

        G_W2 = torch.tensor(xavier_init([H_Dim1, H_Dim2]),requires_grad=True, device="cuda")
        G_b2 = torch.tensor(np.zeros(shape = [H_Dim2]),requires_grad=True, device="cuda")

        G_W3 = torch.tensor(xavier_init([H_Dim2, Dim]),requires_grad=True, device="cuda")
        G_b3 = torch.tensor(np.zeros(shape = [Dim]),requires_grad=True, device="cuda")
    else:
        G_W1 = torch.tensor(xavier_init([Dim*2, H_Dim1]),requires_grad=True)     # Data + Mask as inputs (Random Noises are in Missing Components)
        G_b1 = torch.tensor(np.zeros(shape = [H_Dim1]),requires_grad=True)

        G_W2 = torch.tensor(xavier_init([H_Dim1, H_Dim2]),requires_grad=True)
        G_b2 = torch.tensor(np.zeros(shape = [H_Dim2]),requires_grad=True)

        G_W3 = torch.tensor(xavier_init([H_Dim2, Dim]),requires_grad=True)
        G_b3 = torch.tensor(np.zeros(shape = [Dim]),requires_grad=True)

    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
    
    optimizer_D = torch.optim.Adam(params=theta_D)
    optimizer_G = torch.optim.Adam(params=theta_G)
    
    #%% 1. Generator
    def generator(new_x,m):
        inputs = torch.cat(dim = 1, tensors = [new_x,m])  # Mask + Data Concatenate
        G_h1 = F.relu(torch.matmul(inputs, G_W1) + G_b1)
        G_h2 = F.relu(torch.matmul(G_h1, G_W2) + G_b2)   
        G_prob = torch.sigmoid(torch.matmul(G_h2, G_W3) + G_b3) # [0,1] normalized Output

        return G_prob

    #%% 2. Discriminator
    def discriminator(new_x, h):
        inputs = torch.cat(dim = 1, tensors = [new_x,h])  # Hint + Data Concatenate
        D_h1 = F.relu(torch.matmul(inputs, D_W1) + D_b1)  
        D_h2 = F.relu(torch.matmul(D_h1, D_W2) + D_b2)
        D_logit = torch.matmul(D_h2, D_W3) + D_b3
        D_prob = torch.sigmoid(D_logit)  # [0,1] Probability Output
        return D_prob

    def discriminator_loss(M, New_X, H):
        # Generator
        G_sample = generator(New_X,M)
        # Combine with original data
        Hat_New_X = New_X * M + G_sample * (1-M)

        # Discriminator
        D_prob = discriminator(Hat_New_X, H)

        #%% Loss
        D_loss = -torch.mean(M * torch.log(D_prob + 1e-8) + (1-M) * torch.log(1. - D_prob + 1e-8))
        return D_loss

    def generator_loss(X, M, New_X, H):
        #%% Structure
        # Generator
        G_sample = generator(New_X,M)

        # Combine with original data
        Hat_New_X = New_X * M + G_sample * (1-M)

        # Discriminator
        D_prob = discriminator(Hat_New_X, H)

        #%% Loss
        G_loss1 = -torch.mean((1-M) * torch.log(D_prob + 1e-8))
        MSE_train_loss = torch.mean((M * New_X - M * G_sample)**2) / torch.mean(M)

        G_loss = G_loss1 + alpha * MSE_train_loss 

        #%% MSE Performance metric
        MSE_test_loss = torch.mean(((1-M) * X - (1-M)*G_sample)**2) / torch.mean(1-M)
        return G_loss, MSE_train_loss, MSE_test_loss

    def test_loss(X, M, New_X):
        #%% Structure
        # Generator
        G_sample = generator(New_X,M)
        #%% MSE Performance metric
        MSE_test_loss = torch.mean(((1-M) * X - (1-M)*G_sample)**2) / torch.mean(1-M)
        return MSE_test_loss, G_sample
    
    # tqdm error s
    
    for it in tqdm(range(iters)):    
    
        #%% Inputs
        mb_idx = sample_idx(Train_No, mb_size)
        X_mb = trainX[mb_idx,:]  

        Z_mb = sample_Z(mb_size, Dim) 
        M_mb = trainM[mb_idx,:]  
        H_mb1 = sample_M(mb_size, Dim, 1-p_hint)
        H_mb = M_mb * H_mb1

        New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce
    
        if use_gpu is True:
            X_mb = torch.tensor(X_mb, device="cuda")
            M_mb = torch.tensor(M_mb, device="cuda")
            H_mb = torch.tensor(H_mb, device="cuda")
            New_X_mb = torch.tensor(New_X_mb, device="cuda")
        else:
            X_mb = torch.tensor(X_mb)
            M_mb = torch.tensor(M_mb)
            H_mb = torch.tensor(H_mb)
            New_X_mb = torch.tensor(New_X_mb)
    
        optimizer_D.zero_grad()
        D_loss_curr = discriminator_loss(M=M_mb, New_X=New_X_mb, H=H_mb)
        D_loss_curr.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = generator_loss(X=X_mb, M=M_mb, New_X=New_X_mb, H=H_mb)
        G_loss_curr.backward()
        optimizer_G.step()  
            
        #%% Intermediate Losses
        if it % 5000 == 0:
            print('Iter: {}'.format(it))
            print('Train_loss: {:.4}'.format(np.sqrt(MSE_train_loss_curr.item())))
            print('Test_loss: {:.4}'.format(np.sqrt(MSE_test_loss_curr.item())))
            print() 
    
    #testset으로 찍어보자
    Z_mb = sample_Z(Test_No, Dim) 
    M_mb = testM
    X_mb = testX

    New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce

    if use_gpu is True:
        X_mb = torch.tensor(X_mb, device='cuda')
        M_mb = torch.tensor(M_mb, device='cuda')
        New_X_mb = torch.tensor(New_X_mb, device='cuda')
    else:
        X_mb = torch.tensor(X_mb)
        M_mb = torch.tensor(M_mb)
        New_X_mb = torch.tensor(New_X_mb)

    MSE_final, Sample = test_loss(X=X_mb, M=M_mb, New_X=New_X_mb)

    print('Final Test RMSE: ' + str(np.sqrt(MSE_final.item())))
    print()
     
    # 결측치 보간!!
    Z_mb = sample_Z(len(val_data), Dim) #Test_No: test_set size, Dim은 feature 수 가변적
    M_mb = np.where(val_data != 0, 1, 0) # NULL 값 여부 표시 마스크
    X_mb = val_data # 실제 결측치 포함 데이터
    M_mb = M_mb.astype(np.float64)
    New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce
    
    if use_gpu is True:
        X_mb = torch.tensor(X_mb, device='cuda')
        M_mb = torch.tensor(M_mb, device='cuda')
        New_X_mb = torch.tensor(New_X_mb, device='cuda')
    else:
        X_mb = torch.tensor(X_mb)
        M_mb = torch.tensor(M_mb)
        New_X_mb = torch.tensor(New_X_mb)

    MSE_real, Sample = test_loss(X=X_mb, M=M_mb, New_X=New_X_mb)

    print('!!!Real Imp Data RMSE: ' + str(np.sqrt(MSE_real.item())))
    print()
    
    # 보간 데이터
    imputed_data = M_mb * X_mb + (1-M_mb) * Sample
#     print("Imputed test data:")

#     if use_gpu is True:
#         print(imputed_data.cpu().detach().numpy())
#     else:
#         print(imputed_data.detach().numpy())
    return np.sqrt(MSE_real.item()), Sample, renormal(Data), renormal(imputed_data.cpu().detach().numpy())
