from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set(xlim=[1, 10], ylim=[0, 50],
       ylabel='Consume-Time(min)', xlabel='Video-Numbers')


ax.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [454/60, 755/60, 1005/60, 1325/60, 1578/60, 1843/60, 1902/60, 2205/60, 2495/60, 2789/60],
        label='1 GPU Server',
        color='g')

ax.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [479/60, 624/60, 832/60, 1004/60, 932/60, 1118/60, 1348/60, 1467/60, 1601/60, 1700/60],
        label='2 GPU Server',
        color='b')

ax.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [436/60, 444/60, 564/60, 705/60, 839/60, 916/60, 1123/60, 1237/60, 1375/60, 1498/60],
        label='3 GPU Server',
        color='r')

ax.legend()
plt.savefig('/home/huyibo-21/result/multi_gpu.pdf', bbox_inches='tight')
# plt.show()