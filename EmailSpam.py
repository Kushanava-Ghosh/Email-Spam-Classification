import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt
from subprocess import call
from dataset_creation import Datacret
#github linked


def create_data(path):
    data = genfromtxt(path, delimiter=',')
    data = np.delete(data, 0, axis=1)
    data = np.delete(data, 0, axis=0)
    return data



def create_Y(array):
    Y = array[:, array.shape[1]-1].reshape((array.shape[0], 1))
    array = np.delete(array, array.shape[1] - 1, axis=1)
    array[array != 0] = 1
    return array, Y



def BEM(train_path, valid_path, test_path, output_valid_BEM, efficency_valid_BEM, output_test_BEM, efficency_test_BEM):
    """ okay so this is EMAIL SPAM CLASSIFIER 
        train_path contains the training dataset
        valid_path contains the validation dataset
        test_path contains the test dataset
        output_test_Bem is the output of the Bernoulli event model
        efficnecy_Bem is the efficency on the test dataset for the bernoulli event model
    """

# Data creation in accordance with the Question



    Train_data = create_data(train_path)
    Valid_data = create_data(valid_path)
    Test_data = create_data(test_path)
    Train_data, y_train = create_Y(Train_data)
    Valid_data, y_valid = create_Y(Valid_data)
    Test_data, y_test = create_Y(Test_data)


    clf = SPAMCLassifier1()
    clf.Parameter_train1(Train_data, y_train)
    predictions = clf.predict1(Valid_data)
    count_FP = 0
    count_TP = 0 
    count_FN = 0
    count_TN = 0
    for i in range(predictions.shape[0]):
        if y_valid[i] == 1 and predictions[i] == 1 :
            count_TP += 1
        elif y_valid[i] == 0 and predictions[i] == 0:
            count_TN += 1
        elif predictions[i] == 1 and y_valid[i] == 0:
            count_FP += 1
        elif predictions[i] == 0 and y_valid[i] == 1:
            count_FN += 1
    precision = (count_TP)*100/(count_TP+count_FP)
    recall = (count_TP)*100/(count_TP+count_FN)
    beta = 0.5
    f_beta_sq = beta*precision*recall/(precision+recall)
    items = ['The Accuracy of MEM model for SPam classifier is ',str((count_TP+count_TN)*100/y_valid.shape[0]+1),' The Precision of MEM model for SPam classifier is ',str(precision) ,' The Recall of MEM model for SPam classifier is ',str(recall),' The F_beta_sqaure score for Spam classifier is ',str(f_beta_sq) ]
    file = open('BEM_items.txt','w')
    file.writelines(items)
    file.close()


def MEM(train_path, valid_path, test_path, output_valid_MEM, efficency_valid_MEM, output_test_MEM, efficency_test_MEM):
    
    """ okay so this is EMAIL SPAM CLASSIFIER 
        train_path contains the training dataset
        valid_path contains the validation dataset
        test_path contains the test dataset
        output_test_Bem is the output of the Bernoulli event model
        efficnecy_Bem is the efficency on the test dataset for the bernoulli event model
    """

# Data creation in accordance with the Question

    Train_data = create_data(train_path)
    Valid_data = create_data(valid_path)
    Test_data = create_data(test_path)
    Train_data, y_train = create_Y(Train_data)
    Valid_data, y_valid = create_Y(Valid_data)
    Test_data, y_test = create_Y(Test_data)



    clf = SPAMClassifier2(ArrJ_Y0=np.zeros((Train_data.shape[1],1)),ArrJ_Y1=np.zeros((Train_data.shape[1],1)))
    clf.Parameter_train2(Train_data, y_train)
    predictions = clf.predict2(Valid_data, y_valid)
    count_FP = 0
    count_TP = 0 
    count_FN = 0
    count_TN = 0
    for i in range(predictions.shape[0]):
        if y_valid[i] == 1 and predictions[i] == 1 :
            count_TP += 1
        elif y_valid[i] == 0 and predictions[i] == 0:
            count_TN += 1
        elif predictions[i] == 1 and y_valid[i] == 0:
            count_FP += 1
        elif predictions[i] == 0 and y_valid[i] == 1:
            count_FN += 1
    precision = (count_TP)*100/(count_TP+count_FP)
    recall = (count_TP)*100/(count_TP+count_FN)
    beta = 0.5
    f_beta_sq = beta*precision*recall/(precision+recall)
    items = ['The Accuracy of MEM model for SPam classifier is ',str((count_TP+count_TN)*100/y_valid.shape[0]+1),' The Precision of MEM model for SPam classifier is ',str(precision) ,' The Recall of MEM model for SPam classifier is ',str(recall),' The F_beta_sqaure score for Spam classifier is ',str(f_beta_sq) ]
    file = open('MEM_items.txt','w')
    file.writelines(items)
    file.close()


def main(train_path, valid_path, test_path, output_valid_BEM, efficency_valid_BEM, output_test_BEM, efficency_test_BEM,output_valid_MEM, efficency_valid_MEM, output_test_MEM, efficency_test_MEM):
    BEM(train_path, valid_path, test_path, output_valid_BEM, efficency_valid_BEM, output_test_BEM, efficency_test_BEM)
    MEM(train_path, valid_path, test_path, output_valid_MEM, efficency_valid_MEM, output_test_MEM, efficency_test_MEM)


class SPAMCLassifier1():
    """
    SPAMClassifier with Bernoulli event model

    """

    def __init__(self, ArrJ_Y0=None, ArrJ_Y1=None, FHiY=None):
        self.ArrJ_Y0 = ArrJ_Y0
        self.ArrJ_Y1 = ArrJ_Y1
        self.FHiY = FHiY

    def Y1(self, y):
        sum1 = 0
        FHiY= 0
        for i in range(y.shape[0]):
            if y[i] == 1:
                sum1 += 1
        return sum1 + 2, y.shape[0]+1-sum1+2

    def FHi_J0(self, x, y, j):
        fhi = 1
        for i in range(x.shape[0]):
            if x[i][j] == 1 and y[i] == 0:
                fhi += 1
        return fhi

    def FHi_J1(self, x, y, j):
        fhi = 1
        for i in range(x.shape[0]):
            if x[i][j] == 1 and y[i] == 1:
                fhi += 1
        return fhi

    def Parameter_train1(self, x, y):
        # This function will solve the paramter values of Bernoulli event Model
        self.FHiY = 0
        self.ArrJ_Y0 = np.zeros((x.shape[1], 1))
        self.ArrJ_Y1 = np.zeros((x.shape[1], 1))
        Y1, Y0 = self.Y1(y)
        self.FHiY = Y1
        for j in range(x.shape[1]):
            self.ArrJ_Y0[j] = self.FHi_J0(x, y, j)/Y0
            self.ArrJ_Y1[j] = self.FHi_J1(x, y, j)/Y1
            print("iterations round 1 ->",j)
        print(Y1,Y0,self.ArrJ_Y0,"okay",self.ArrJ_Y1)

    def predict1(self, x):
        predict = np.zeros((x.shape[0], 1))
        
        for i in range(x.shape[0]):
            value1 = 1.0
            value0 = 1.0
            for j in range(x.shape[1]):
                value1 = value1 + np.log(((self.ArrJ_Y1[j][0])**(x[i][j]))*((1 - self.ArrJ_Y1[j][0])**(1-x[i][j])))
                value0 = value0 + np.log(((self.ArrJ_Y0[j][0])**(x[i][j]))*((1 - self.ArrJ_Y0[j][0])**(1-x[i][j])))
            
            if value0 + np.log(self.FHiY) < value1 + np.log(self.FHiY):
                predict[i][0] = 1
            else:
                predict[i][0] = 0
            print("iterations round 2 ->",i)
        return predict

class SPAMClassifier2():
    """
    This the spamclassfier with the Multinomial event model
    """

    def __init__(self, ArrJ_Y0=None, ArrJ_Y1=None, FHiY=0):
        self.ArrJ_Y0 = ArrJ_Y0
        self.ArrJ_Y1 = ArrJ_Y1
        self.FHiY = FHiY

    def Y_1count(self, y):
        count = 0
        for i in range(y.shape[0]):
            if y[i][0] == 1:
                count += 1
        return count/(y.shape[0]+1)

    def count_Y(self, x, y):
        sum_x = np.sum(x, axis=1)
        count1 = 0
        count0 = 0
        # print(sum_x)
        for i in range(y.shape[0]):
            if y[i][0] == 1:
                count1 += sum_x[i]
            else:
                count0 += sum_x[i]
        count1 = count1 + x.shape[1]
        count0 = count0 + x.shape[1]
        return count0, count1

    def Arry_0_1(self, x, y):
        Y0, Y1 = self.count_Y(x, y)
        for i in range(x.shape[0]):
            if y[i] == 0:
                for j in range(x.shape[1]):
                    self.ArrJ_Y0[j][0] += x[i][j]
            else:
                for j in range(x.shape[1]):
                    self.ArrJ_Y1[j][0] += x[i][j]
            print(i)

        self.ArrJ_Y0 = self.ArrJ_Y0/Y0
        self.ArrJ_Y1 = self.ArrJ_Y1/Y1

    def Parameter_train2(self, x, y):
        self.Arry_0_1(x, y)
        self.FHiY = self.Y_1count(y)

    def predict2(self, x, y):
        predictions  = np.zeros((x.shape[0],1))
        for i in range(x.shape[0]):
            # print(i)
            value0 = 1
            value1 = 1
            for j in range(self.ArrJ_Y1.shape[0]):
                if x[i][j] != 0:
                    value0 = value0 + np.log(pow(self.ArrJ_Y0[j][0],x[i][j]))
                    value1 = value1 + np.log(pow(self.ArrJ_Y1[j][0],x[i][j]))
            print("Iteration number -> ",i)
            if value0*self.FHiY > value1*self.FHiY:
                predictions[i][0] = 0
            else:
                predictions[i][0] = 1
        return predictions    


if __name__ == '__main__':
    #1st argument -> Fraction of data to be training dataset , 2nd argument is -> Fraction of data to be validation dataset
    Datacret(0.7,0.2)
    main(train_path='train.csv',
         valid_path='valid.csv',
         test_path='test.csv',
         output_valid_BEM='Bem_valid_output.txt',
         efficency_valid_BEM='Bem_eff.txt',
         output_test_BEM='Bem_test_output.txt',
         efficency_test_BEM='Bem_eff.txt',
         output_valid_MEM='Mem_valid_output.txt',
         efficency_valid_MEM='Mem_eff.txt',
         output_test_MEM='Mem_test_output.txt',
         efficency_test_MEM='Mem_eff.txt')
