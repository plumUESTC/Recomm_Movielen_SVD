import numpy as np
import random

res=[]
res2=[]
class SVD_plus_plus:
    def __init__(self,Matrix,K=20):
        self.matrix=np.array(Matrix)
        self.K=K
        self.b_i={}
        self.b_u={}
        self.q_i={}
        self.p_u={}
        self.Aveg=np.mean(self.matrix[:,2])
        self.y_={}
        self.Dic_u={}
        for i in range(self.matrix.shape[0]):
            User_id=self.matrix[i,0]
            Item_id=self.matrix[i,1]
            self.Dic_u.setdefault(User_id,[])
            self.Dic_u[User_id].append(Item_id)
            self.b_i.setdefault(Item_id,0)
            self.b_u.setdefault(User_id,0)
            self.q_i.setdefault(Item_id,np.random.random((self.K,1))/10*np.sqrt(self.K))
            self.p_u.setdefault(User_id,np.random.random((self.K,1))/10*np.sqrt(self.K))
            self.y_.setdefault(Item_id,np.zeros((self.K,1))+.1)

    def Prediction(self,User_id,Item_id): 
        self.b_i.setdefault(Item_id,0)
        self.b_u.setdefault(User_id,0)
        self.q_i.setdefault(Item_id,np.zeros((self.K,1)))
        self.p_u.setdefault(User_id,np.zeros((self.K,1)))
        self.y_.setdefault(User_id,np.zeros((self.K,1)))
        self.Dic_u.setdefault(User_id,[])
        Extra_User_item,SqrtNu=self.Get_extra(User_id, Item_id)
        Rating=float(self.Aveg+self.b_i[Item_id]+self.b_u[User_id]+np.sum(self.q_i[Item_id]*(self.p_u[User_id]+Extra_User_item))) 
        #对于高于5分或者低于1分的，统一处理
        if Rating>5:
            Rating=5
        if Rating<1:
            Rating=1
        return Rating

    def Get_extra(self,User_id,Item_id):
        N_U=self.Dic_u[User_id]
        Ieng_NU=len(N_U)
        SqrtNU=np.sqrt(Ieng_NU)
        y_u=np.zeros((self.K,1))
        if Ieng_NU==0:
            Extra_User_Item=y_u
        else:
            for i in N_U:
                y_u+=self.y_[i]
            Extra_User_Item = y_u / SqrtNU
        
        return Extra_User_Item,SqrtNU

    
    def Training(self,steps=200000,Gamma=0.1,Lambda=0.1):    
        for step in range(steps):
            print('Step',step+1,'is Running Now!')
            Rand_k=np.random.permutation(self.matrix.shape[0]) 
#            RMSE=0.0
            MAE=0.0
            for i in range(self.matrix.shape[0]):
                clc=Rand_k[i]
                User_id=self.matrix[clc,0]
                Item_id=self.matrix[clc,1]
                Rating=self.matrix[clc,2]
                predict=self.Prediction(User_id, Item_id)
                Extra_User_Item,Sqrt_NU=self.Get_extra(User_id, Item_id)
                E_ui=Rating-predict
                MAE=MAE+abs(E_ui)
#                RMSE+=E_ui**2
                self.b_u[User_id]+=Gamma*(E_ui-Lambda*self.b_u[User_id])  
                self.b_i[Item_id]+=Gamma*(E_ui-Lambda*self.b_i[Item_id])
                self.p_u[User_id]+=Gamma*(E_ui*self.q_i[Item_id]-Lambda*self.p_u[User_id])
                self.q_i[Item_id]+=Gamma*(E_ui*(self.p_u[User_id]+Extra_User_Item)-Lambda*self.q_i[Item_id])
                for clc in self.Dic_u[User_id]:
                    self.y_[clc]+=Gamma*(E_ui*self.q_i[clc]/Sqrt_NU-Lambda*self.y_[clc])
                                    
            Gamma=0.93*Gamma
            print('MAE : ',MAE/self.matrix.shape[0])
            res2.append(MAE/self.matrix.shape[0])
#            print('RMSE: ',np.sqrt(RMSE/self.matrix.shape[0]))
#            res.append(np.sqrt(RMSE/self.matrix.shape[0]))
            if(step+1>=2):
                if(abs(res2[-1]-res2[-2])<=0.0001):break; # 等到收敛即退出
    
    def TEST(self,TestData):
        TestData=np.array(TestData)
        MAE=0.0
        for i in range(TestData.shape[0]):
            User_id=TestData[i,0]
            Item_id=TestData[i,1]
            Rating=TestData[i,2]
            E_ui=Rating-self.Prediction(User_id, Item_id)
#            RMSE+=E_ui**2
            MAE=MAE+abs(E_ui)
        print('MAE of TestData :',MAE/TestData.shape[0])
#        print('RMSE of TestData : ',np.sqrt(RMSE/TestData.shape[0]))
    
    
def Get_init_Data(): 
    import re
    f=open("C:/Users/z1325/Desktop/u1.base",'r',encoding='ISO-8859-1')
    lines=f.readlines()
    f.close()
    data=[]
    for line in lines:
        list=re.split('\t|\n',line)
        if int(list[2]) !=0:
            data.append([int(i) for i in list[:3]])
            
    TrainingData=data
    f=open("C:/Users/z1325/Desktop/u1.test",'r',encoding='ISO-8859-1')
    lines=f.readlines()
    f.close()
    data=[]
    for line in lines:
        list=re.split('\t|\n',line)
        if int(list[2]) !=0:
            data.append([int(i) for i in list[:3]])
            
    TestData=data
    
    return TrainingData,TestData
    
TrainingData,TestData=Get_init_Data()
a=SVD_plus_plus(TrainingData,30)  
a.Training()
a.TEST(TestData)
print(res2)
        
                 