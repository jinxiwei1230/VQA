from matplotlib import pyplot as plt

x=[1,2,3,4,5]  # 确定柱状图数量,可以认为是x方向刻度
y=[5,7,4,3,1]  # y方向刻度

color=['red','black','peru','orchid','deepskyblue']
x_label=['face','object','scene','length','clarity']
plt.xticks(x, x_label)  # 绘制x刻度标签
plt.bar(x, y, color=color)  # 绘制y刻度标签

#设置网格刻度
# plt.grid(True,linestyle=':',color='r',alpha=0.6)
plt.show()