#..Data Mining and Visualisation..#
#..Course Work 1..#

#importing required libraries
import numpy as np
import random 
class Data(object): 
    
    def __init__(self,fileName):
       
        self.fileData = open(fileName).read().splitlines()
        self.data=[]

        random.seed(2)
        random.shuffle(self.fileData)

        temp=[]
        for i,j in enumerate(self.fileData):
            split=j.split(',class-') 

            x=np.array(split[0].split(',')).astype(float) 
        
            y=split[1]
            if y not in temp:
                np.append(temp,y) 
            self.data.append({'x': x, 'class-id': y})
        
        self.row = len(self.data[0]["x"]) 

class Perceptron(object):
    
    def __init__(self, positive, negative=None,maxIter=20):
        self.positive = positive 
        self.negative = negative 
        self.maxIter=maxIter
    
    def Train(self,D,regularisationFactor = 0):
        weights = np.zeros(D.row) ##Initialising weights as vector of zeros and bias
        bias=1
        y = Class_Labels(D,self.positive,self.negative)
        for j in range(self.maxIter):

            correct,incorrect=0,0 
        for i in range(len(D.data)):
                x = D.data[i]["x"] 
                activation=np.sign(np.dot(weights,x)+bias)

                if y[i]==0: pass
                elif activation==y[i]:
                    correct+=1 
                elif y[i]* activation <= 0: 
                    weights=(1- 2*regularisationFactor)*weights + y[i]*x
                    bias+=y[i]
                    incorrect+=1
                else:
                    incorrect+=1
        self.weights=weights
        self.accuracy=correct/(correct+incorrect)*100 
        return self.accuracy

    def Test(self, D):
        y = Class_Labels(D,self.positive,self.negative)
        correct,incorrect = 0,0

        for i in range(len(D.data)):
            x = D.data[i]['x'] 
            
            activation=np.sign(np.dot(self.weights,x))

            if y[i]==0:pass 
            elif y[i]==activation:
                correct += 1
            else:
                incorrect += 1

        self.accuracy=correct/(correct+incorrect)*100
        return self.accuracy
def Class_Labels(D,positive,negative):
        y = {}
        for i in range(len(D.data)):
            classNum = D.data[i]["class-id"]
            if classNum == positive: 
                y[i] = 1 
            elif negative:
                y[i] = -1 if classNum == negative else 0
            else:y[i] = -1   
        return y
    
    #.. Question 2..#
    #.. Question 3..#

def main(): 
    print("-------------Question 2 and 3-------------------")
    
    Train_data = Data("train.data")
    Class_1 = Perceptron("1","2")
    Class_2 = Perceptron("2","3")
    Class_3 = Perceptron("1","3")
    
    print("Training Perceptron")
    Class_1.Train(Train_data)
    Class_2.Train(Train_data)
    Class_3.Train(Train_data)
    train=[Class_1,Class_2,Class_3]
    for i in train:
        print("Training Accuracy rate:%.2f%%"%i.accuracy)

    test_data = Data("test.data")
    print("\nTesting data")
    Class_1.Test(test_data)
    Class_2.Test(test_data)
    Class_3.Test(test_data)
    for i in train:
        print("Testing Accuracy rate:%.2f%%"%i.accuracy)

    print("-----------------------------------------------")
    
    #..Question 4..#

    print("----------------Question 4---------------------")
    Train_data = Data("train.data")
    
    Class_1 = Perceptron("1")
    Class_2 = Perceptron("2")
    Class_3 = Perceptron("3")
    
    print("Training Perceptron")
    Class_1.Train(Train_data)
    Class_2.Train(Train_data)
    Class_3.Train(Train_data)

    train=[Class_1,Class_2,Class_3]
    for i in train:
        print("Training Accuracy rate:%.2f%%"%i.accuracy)

    test_data = Data("test.data")
    print("\nTesting data")
    Class_1.Test(test_data)
    Class_2.Test(test_data)
    Class_3.Test(test_data)
    for i in train:
        print("Testing Accuracy rate:%.2f%%"%i.accuracy)
    print("-----------------------------------------------")
    
    #..Question 5..#

    print("--------------Question 5-----------------------")

    Train_data = Data("train.data")
    regularisation = [0.01, 0.1, 1.0, 10.0, 100.0]
    Class_1 = Perceptron("1")
    Class_2 = Perceptron("2")
    Class_3 = Perceptron("3")

    test_data = Data("test.data")
    print("Testing data")
    for i in (regularisation):
        print("\nRegularisation factor:%.2f\n"%i)
        Class_1.Train(Train_data,i)
        Class_1.Test(test_data)
        print("Testing Accuracy rate:%.2f%%"%Class_1.accuracy)
        
        Class_2.Train(Train_data,i)
        Class_2.Test(test_data)
        print("Testing Accuracy rate:%.2f%%"%Class_2.accuracy)

        Class_3.Train(Train_data,i)
        Class_3.Test(test_data)
        print("Testing Accuracy rate:%.2f%%"%Class_3.accuracy)

    print("-----------------------------------------------")

if __name__ == '__main__':
    main()