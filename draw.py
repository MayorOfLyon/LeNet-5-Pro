import matplotlib.pyplot as plt
from itertools import chain

with open('./CIFAR-100_opt/Adam_100.txt', 'r') as file:
    lines = file.readlines()
total_loss_adam = [line.strip().split() for line in lines]

with open('./CIFAR-100_opt/SGD_100.txt', 'r') as file:
    lines = file.readlines()
total_loss_sgd = [line.strip().split() for line in lines]

with open('./CIFAR-100_opt/Adadelta_100.txt', 'r') as file:
    lines = file.readlines()
total_loss_delta = [line.strip().split() for line in lines]

total_loss_adam = list(chain.from_iterable(total_loss_adam))
total_loss_adam = [round(float(x), 3) for x in total_loss_adam]

total_loss_sgd = list(chain.from_iterable(total_loss_sgd))
total_loss_sgd = [round(float(x), 3) for x in total_loss_sgd]

total_loss_delta = list(chain.from_iterable(total_loss_delta))
total_loss_delta = [round(float(x), 3) for x in total_loss_delta]


iterations = list(range(1, len(total_loss_adam) + 1))

plt.plot(iterations, total_loss_adam, label='Adam')
plt.plot(iterations, total_loss_sgd, label='SGD')
plt.plot(iterations, total_loss_delta, label='Adadelta')

plt.xlabel('Iterations or Epochs')
plt.ylabel('Loss')
plt.title('Loss Visualization for Different Optimizers on CIFAR-10')
plt.legend()  

plt.show()
        