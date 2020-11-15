import numpy as np
import LoadData
import math
import csv
import random

from LoadData import Loading_Data
class Usr:

    def __init__(self, Iteration, num_Factors, num_user, num_item, User_Item,Item_User, num_Bins, num_days, Mintime_Of_sec):
        self.Gamma = 0.01
        self.tau = 0.01
        self.Gamma_alpha = 0.0001
        self.Loss_alpha = 0.0004
        self.MaxTime = num_days
        self.MinTime = 0
        self.MinTime_Of_sec = Mintime_Of_sec
        self.iter = Iteration
        self.User_Item = User_Item
        self.Item_User = Item_User
        self.Fac = num_Factors + 1
        self.nU = num_user
        self.nI = num_item
        self.nB = num_Bins
        self.nD = num_days

        #初始化开始
        print('初始化开始......')
        b_u, b_i, u_f, i_f, y_j, Sigma_yj, bi_bin, alpha_u, b_ut, alpha_uk, FactorUser_t,k1,k2,k3,k4 = self.init(self.nU, self.nI, self.Fac, self.nB)

        self.bu = b_u
        self.bi = b_i
        self.bi_bin = bi_bin
        self.alpha_u = alpha_u
        self.bu_t = b_ut
        self.UserFac = u_f
        self.itemFac = i_f
        self.y_j = y_j
        self.sumYj = Sigma_yj
        self.alpha_uk = alpha_uk
        self.Facuser_t = FactorUser_t
        self.k1 = k1
        self.k2 =k2
        self.k3 = k3
        self.k4 = k4
        print ("初始化结束......")

        self.Aveg = self.Get_average()
        print ("Average = ", self.Aveg)
        print ("Training ......")
        self.Training(self.iter)
        print ("Training over......")

        print ("Caculating ......")
        RMSE = self.RMSE()
        print ("RMSE = ", RMSE)
        print ("Caculating over......")


    def init(self, nUsers, nItems, nFac, nBins):
        bu = np.zeros(nUsers + 1)
        bi = np.zeros(nItems + 1, dtype = 'float64')
        bi_bin = np.zeros((nItems + 1, nBins))
        alpha_u = np.zeros(nUsers + 1, dtype = 'float64')
        bu_t = np.zeros((nUsers + 1, self.nD))
        UserFac = np.random.random((nUsers + 1, nFac))
        ItemFac = np.random.random((nItems + 1, nFac))
        k1 = 1
        k2 = 5
        k3 = 1
        k4 = 5
        y_j = np.zeros((nItems + 1, nFac), np.float64)

        Sigma_yj = np.random.random((nUsers + 1, nFac))
        alpha_u_k = np.zeros((nUsers + 1, nFac))
        UserFac_t = []
        for j in range(nUsers + 1):
            F_day = np.random.random((nFac, self.nD))
            for i in range(len(F_day)):
                for k in range(len(F_day[i])):
                    F_day[i][k] = F_day[i][k]/math.sqrt(20)

            UserFac_t.append(F_day)
        for i in range(len(UserFac)):
            for j in range(len(UserFac[i])):
                UserFac[i][j] = UserFac[i][j]/math.sqrt(20)

        for i in range(len(ItemFac)):
            for j in range(len(ItemFac[i])):
                ItemFac[i][j] = ItemFac[i][j]/math.sqrt(20)

        for i in range(len(Sigma_yj)):
            for j in range(len(Sigma_yj[i])):
                Sigma_yj[i][j] = Sigma_yj[i][j]/math.sqrt(20)
        return bu, bi, UserFac, ItemFac, y_j, Sigma_yj, bi_bin, alpha_u, bu_t, alpha_u_k, UserFac_t ,k1,k2,k3,k4

    def Training(self, iter):
        for i in range(iter):
            print ('..........', i + 1,' .........')
            self.OneIter()
            MAE,RMSE = self.RMSE()
            print ("Iter: ", i + 1, ", MAE = ", MAE)
            print ("Iter: ", i + 1, ", RMSE = ", RMSE)
    def MeantimeI(self,itemID):
        sum = 0
        cnt = 0
        Len = len(self.Item_User[itemID])
        if Len > 0:
            for i in range(Len):
                sum += self.Item_User[itemID][i][2]
                cnt += 1
            return sum/cnt*1.0
        else:
            return 0
    def OneIter(self):
        for User_Id in range(1, len(self.User_Item) + 1):
            tmpe_SumYj = np.zeros(self.Fac, dtype='float')
            len_ui = len(self.User_Item[User_Id])
            if len_ui > 0:
                SQRT = 1/(math.sqrt(len_ui))
                for FAC in range(self.Fac):
                    sum_y = 0
                    for j in range(len_ui):
                        pos_item = self.User_Item[User_Id][j][0]
                        sum_y += self.y_j[pos_item][FAC]
                    self.sumYj[User_Id][FAC] = sum_y
                for it in range(len_ui):
                    Item_id = self.User_Item[User_Id][it][0]
                    Rating = self.User_Item[User_Id][it][1]
                    Timestp = self.User_Item[User_Id][it][2]
                    prediction = self.Prediction(User_Id, Item_id, Timestp)
                    t_i = self.MeantimeI(Item_id)
                    Err = Rating - prediction
                    self.bu[User_Id]+= self.Gamma * (Err - self.tau * self.bu[User_Id])
                    self.bi[Item_id] += self.Gamma * (Err - self.tau * self.bi[Item_id])
                    self.k1 += self.Gamma*(Err*self.bi[Item_id]/(1+math.exp( -self.k2*(Timestp-t_i)/1000 ) )-self.tau*self.k1)
                    self.k2 += self.Gamma*(Err*self.bi[Item_id]*self.k1*(Timestp-t_i)/math.pow( 1+math.exp( -self.k2*(Timestp-t_i)/1000),2) -self.tau*self.k2)
                    self.k3 += self.Gamma*(Err*self.bu[User_Id]*math.exp(-self.k4*(Timestp-t_i)/1000)-self.tau*self.k3)
                    self.k4 += self.Gamma*(-Err*self.bu[User_Id]*self.k3*math.exp(-self.k4*(Timestp-t_i)/1000)*(math.pow((Timestp-t_i)/1000,1)) -self.tau*self.k4)
     #               self.bi_bin[Item_id][self.compute_bin(Timestp)] += self.Gamma * (Err - self.tau* self.bi_bin[Item_id][self.compute_bin(Timestp)])
                    for k in range(self.Fac):
                        u_f = self.UserFac[User_Id][k]
                        i_f = self.itemFac[Item_id][k]
                        u_F_t = self.Facuser_t[User_Id][k][Timestp]
                        self.UserFac[User_Id][k] += self.Gamma * (Err * i_f - 0.015 * u_f)
                        self.itemFac[Item_id][k] += self.Gamma * (Err * (u_f + SQRT * self.sumYj[User_Id][k] + self.alpha_uk[User_Id][k]*self.dev(User_Id,Timestp)) - 0.015 * i_f)
     #                   self.alpha_uk[User_Id][k] += self.Gamma_alpha * (Err * self.dev(User_Id, Timestp) * i_f - self.Loss_alpha * self.alpha_uk[User_Id][k])
                        self.Facuser_t[User_Id][k][Timestp] += self.Gamma * (Err * i_f - 0.015 * u_F_t)
                        tmpe_SumYj[k] += Err * SQRT * i_f

                for j in range(len_ui):
                    ItID = self.User_Item[User_Id][j][0]
                    for FAC in range(self.Fac):
                        tmp_Yj = self.y_j[ItID][FAC]
                        self.y_j[ItID][FAC] += self.Gamma * (tmpe_SumYj[FAC] - 0.015 * tmp_Yj)
                        self.y_j[ItID][FAC] = round(self.y_j[ItID][FAC], 4)
                        self.sumYj[User_Id][FAC] += self.y_j[ItID][FAC] - tmp_Yj

        for User_Id in range(1, len(self.User_Item) + 1):
            len_ui = len(self.User_Item[User_Id])
            if len_ui > 0:
                SQRT = 1/(math.sqrt(len_ui))
                for k in range(self.Fac):
                    sum_yj = 0
                    for i in range(len_ui):
                        ItID = self.User_Item[User_Id][i][0]
                        sum_yj += self.y_j[ItID][k]
                    self.sumYj[User_Id][k] = sum_yj

        self.Gamma *= 0.9
#        self.Gamma_alpha *= 0.9
    def Get_average(self):
        sum = 0
        cnt = 0
        l = len(self.User_Item)
        for i in range(1, l + 1):
            Leng = len(self.User_Item[i])
            for j in range(Leng):
                sum += self.User_Item[i][j][1]
                cnt += 1
        Aveg = sum/cnt*1.0
        return Aveg
        #计算桶的索引
    def compute_bin(self, DayOfRating):
        Gap = (self.MaxTime - self.MinTime) / self.nB
        Bin_index = np.minimum(self.nB - 1, int((DayOfRating - self.MinTime)/Gap))
        return Bin_index
    # dev 函数
    def dev(self, userID, Time):
        Devia = np.sign(Time - self.MeanTime(userID)) * pow(abs(Time - self.MeanTime(userID)), 0.015)
        return Devia
    # 计算平均时间
    def MeanTime(self, userID):
        sum = 0
        cnt = 0
        Len = len(self.User_Item[userID])
        if Len > 0:
            for i in range(Len):
                sum += self.User_Item[userID][i][2]
                cnt += 1
            return sum/cnt*1.0
        else:
            return 0



    def Prediction(self, User, Item, DayIndex):
        Len = len(self.User_Item[User])
        t_i = self.MeanTime(User)
        if Len > 0:
            SQRT = 1/ math.sqrt(Len)
        else:
            SQRT = 0
        tmp = 0
        for k in range(self.Fac):
            tmp += ((self.UserFac[User][k] + self.alpha_uk[User][k] * self.dev(User, DayIndex) + self.Facuser_t[User][k][DayIndex]) + (SQRT * self.sumYj[User][k])) * self.itemFac[Item][k]
#        prediction = self.Aveg + self.bu[User] + self.bi[Item] + self.bi_bin[Item][self.compute_bin(DayIndex)]+ self.alpha_u[User] * self.dev(User,DayIndex) + self.bu_t[User][DayIndex] + tmp
        prediction = self.Aveg + self.bu[User] + self.bi[Item]*self.k1/(1+math.exp(-self.k2*(DayIndex-t_i)/1000)) +   self.bu_t[User][DayIndex] + tmp + self.bu[User]*self.k3*math.exp(-self.k4*(DayIndex-t_i)/1000)

        return prediction

    def RMSE(self):
        with open("C:/Users/z1325/Desktop/u1.test", 'r',encoding='ISO-8859-1') as f:
            Data = csv.reader(f, delimiter = '\t')
            MSE = 0
            MAE = 0
            cnt = 0
            for row in Data:
                userid = int(row[0])
                itemid = int(row[1])
                Rating = float(row[2])
                tmp = int(row[3])
                day = min(self.nD - 1, int((tmp - self.MinTime_Of_sec)/86400))
                predict = self.Prediction(userid, itemid, day)
                MAE = MAE + abs(Rating - predict)
                MSE += math.pow((Rating - predict), 2)
                cnt += 1
            MAE = MAE/cnt*1.0
            MSE = math.sqrt(MSE/cnt*1.0)
            return MAE,MSE

lm =Loading_Data()
userItems,Item_User, n_User, n_Item, nDays, Mintimestp = lm.MAIN()
nFac = 20
nBin = 6
MOV = Usr(30, nFac, n_User, n_Item, userItems,Item_User, nBin, nDays, Mintimestp)
