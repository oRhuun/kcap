import random as random
if __name__ == '__main__':
    x=[]
    y=[]
    for n in range(0,50):
        x.insert(n, random.randint(0,100))
        y.insert(n, random.randint(0,100))
        
    print(x)
    print(y)
    f = open("testData.txt", "a")
    f.write(str(x))
    f.write(str(y))
    f.close()