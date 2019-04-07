# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
#X轴，Y轴数据
one_thread_time = 660.928
x = [2, 4, 8, 12, 16, 20, 24]
y = [329.606, 162.836, 96.983, 105.121, 67.375, 55.546, 45.490]
for i in range(len(y)):
    y[i] = one_thread_time /y[i] 
    print(y[i])
plt.figure(figsize=(8,4)) #创建绘图对象
plt.plot(x,y,"b--",linewidth=2)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
plt.xlabel("threads") #X轴标签
plt.ylabel("speedup")  #Y轴标签
plt.title("speed up") #图标题
plt.show()  #显示图