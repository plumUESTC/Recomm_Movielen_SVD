import numpy as np
import math
import csv
import gzip
import matplotlib.pyplot as plt
from collections import OrderedDict
from operator import itemgetter
import struct
import collections
import numpy as np
import matplotlib as mp
import random
import math
import matplotlib.pyplot as plt



class Loading_Data:
    def Init_TrainingData(self, PATH):
         with open(PATH, 'rb') as f:
            Matrix = []
            User_Items = {}
            Item_Users = {}
            Maximum_Items_n = []
            Time_Stamp = []
            
            for line in f:
                row=[]
                a = line.decode().split('\t')

                User = int(a[0])
                Item = int(a[1])
                Rating = float(a[2])
                Time_stp = int(a[3])
                row.append(User)
                row.append(Item)
                row.append(Rating)
                row.append(Time_stp)

                Matrix.append(row)
                Maximum_Items_n.append(Item)
                Time_Stamp.append(Time_stp)
                # 记录每个用户和物品的信息
                if User not in User_Items:
                    User_Items[User] = [(Item, Rating, Time_stp)]
                else:
                    if Item not in User_Items[User]:
                        User_Items[User].append((Item, Rating, Time_stp))

                if Item not in Item_Users:
                    Item_Users[Item] = [(User, Rating, Time_stp)]
                else:
                    if User not in Item_Users[Item]:
                        Item_Users[Item].append((User, Rating, Time_stp))
            Min_timestp = min(Time_Stamp)
            Max_timestp = max(Time_Stamp)# 记录最大和最小的时间戳
            return Matrix, User_Items, Item_Users, Min_timestp, Max_timestp

    def Days_number(self, Min_timestp, Max_timestp):
        N_Days = int((Max_timestp - Min_timestp)/86400)#转化为天数 24*60*60
        return N_Days

    def CountDay(self, TimeStamp, Days_num,Min_timestp):
        Dayindex = np.minimum(Days_num - 1, int((TimeStamp - Min_timestp)/86400))
        return Dayindex

    def Turn_Timestp_to_Day(self, Matrix, NsDays, Min_timestp):
        User_Item = {}
        Item_User = {}
        Item_User.setdefault(599,[3])
        for i in range(len(Matrix)):
            user = Matrix[i][0]
            item = Matrix[i][1]
            Rating = Matrix[i][2]
            timestp = Matrix[i][3]
            Dayindex = self.CountDay(timestp, NsDays, Min_timestp)
            #记录用户和物品信息
            if user not in User_Item:
                User_Item[user] = [(item, Rating, Dayindex)]
            else:
                if item not in User_Item[user]:
                    User_Item[user].append((item, Rating, Dayindex))

            if item not in Item_User:
                Item_User[item] = [(user, Rating, Dayindex)]
            else:
                if user not in Item_User[item]:
                    Item_User[item].append((user, Rating, Dayindex))
        return User_Item, Item_User

    def MAIN(self):
        Mat, userItems, itemUsers, Min_timestp, Max_timestp = self.Init_TrainingData("C:/Users/z1325/Desktop/u1.base")
        days_num = self.Days_number(Min_timestp, Max_timestp)
        extr_User_Item, extr_Item_User = self.Turn_Timestp_to_Day(Mat,days_num, Min_timestp)
        len_Users = len(extr_User_Item)
        len_Items = 1683
        Spand = Max_timestp - Min_timestp

        return extr_User_Item,extr_Item_User, len_Users, len_Items, days_num, Min_timestp


