from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set(xlim=[119, 604], ylim=[0, 65],
       ylabel='Consume-Time(min)', xlabel='Video-Length(s)')
ax.plot([119, 183, 240, 299, 360, 419, 481, 539, 604],
        [453/60, 723/60, 978/60, 1353/60, 1663/60, 2033/60, 2119/60, 2336/60, 2398/60],
        label='No Parallelization(540p)',
        color='g')
ax.plot([119, 183, 240, 299, 360, 419, 481, 539, 604],
        [530/60, 824/60, 1144/60, 1564/60, 1919/60, 2046/60, 2640/60, 2680/60, 2952/60],
        label='No Parallelization(720p)',
        color='b')
ax.plot([119, 183, 240, 299, 360, 419, 481, 539, 604],
        [739/60, 1132/60, 1500/60, 1999/60, 2429/60, 2923/60, 3105/60, 3413/60, 3718/60],
        label='No Parallelization(1080p)',
        color='r')
ax.plot([119, 183, 240, 299, 360, 419, 481, 539, 604],
        [73/60, 58/60, 189/60, 178/60, 220/60, 270/60, 459/60, 464/60, 479/60],
        label='Parallelization(540)',
        color='g',
        linestyle=':')
ax.plot([119, 183, 240, 299, 360, 419, 481, 539, 604],
        [93/60, 102/60, 91/60, 134/60, 332/60, 326/60, 571/60, 538/60, 567/60],
        label='Parallelization(720p)',
        color='b',
        linestyle=':')
ax.plot([119, 183, 240, 299, 360, 419, 481, 539, 604],
        [198/60, 263/60, 284/60, 365/60, 436/60, 559/60, 780/60, 765/60, 794/60],
        label='Parallelization(1080p)',
        color='r',
        linestyle=':')


ax.legend()
# plt.savefig('/home/huyibo-21/result/one_gpu.pdf', bbox_inches='tight')
plt.show()