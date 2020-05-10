from matplotlib import pyplot as plt

filename = '20000_gen_c.txt'

l2_loss = []

f = open(filename, 'r')
line = True
while line:
    line = f.readline()
    if line == '':
        break
    split = line.split()
    l2 = split[3]
    l2_loss.append(float(l2[:-1]))    
f.close()

plt.figure(figsize=(8, 8))
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('L2 Loss', fontsize=20)
plt.plot(l2_loss)
plt.savefig('{}.png'.format(filename), dpi=100)
