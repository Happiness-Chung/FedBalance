import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

f1 = open('C:/Users/hb/Desktop/code/3.FedBalance_mp/logs/CheXpert_fedavg_acc.txt', 'r')
f2 = open('C:/Users/hb/Desktop/code/3.FedBalance_mp/logs/CheXpert_fedprox_acc.txt', 'r')
f3 = open('C:/Users/hb/Desktop/code/3.FedBalance_mp/logs/CheXpert_moon_acc.txt', 'r')
f4 = open('C:/Users/hb/Desktop/code/3.FedBalance_mp/logs/CheXpert_fedalign_acc.txt', 'r')
f5 = open('C:/Users/hb/Desktop/code/3.FedBalance_mp/logs/CheXpert_fedbalance_acc.txt', 'r')

accs1 = f1.readlines()
accs2 = f2.readlines()
accs3 = f3.readlines()
accs4 = f4.readlines()
accs5 = f5.readlines()

acc1 = []
acc2 = []
acc3 = []
acc4 = []
acc5 = []

for i in range(len(accs1)):
    acc1.append(float(accs1[i].strip()))
    acc2.append(float(accs2[i].strip()))
    acc3.append(float(accs3[i].strip()))
    acc4.append(float(accs4[i].strip()))
    acc5.append(float(accs5[i].strip()))

plt.plot(acc1, label='FedAvg')
plt.plot(acc2, label='FedProx')
plt.plot(acc3, label='MOON')
plt.plot(acc4, label='FedAlign')
plt.plot(acc5, label='FedBalance')
plt.xticks([5,10,15,20,25,30,35,40,45,50])
plt.title('CheXpert')
plt.xlabel('Communication Round')
plt.ylabel('AUC')
plt.legend()

plt.savefig('C:/Users/hb/Desktop/code/3.FedBalance_mp/logs/CheXpert_AUC.png')