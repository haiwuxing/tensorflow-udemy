# Simple regresstion example by Li Jian
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import os
# windows 系统加上当前目录，其他系统默认目录为当前目录
# windows 的默认目录为 C:\Users\UserName\
dir = ''
if (os.name == 'nt'):
    scriptpath = os.path.realpath(__file__)
    dir = os.path.dirname(scriptpath)

x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
m = 0.3
b = 0.6
y_label = x_data*m + b

print('m=', m, 'b=',b)
#print('y_label=',y_label)
plt.plot(x_data,y_label)
# windows 上必须加这句，jupyter notebook 上不需要
#plt.show()

# 随机值，1.0，1.0
m_tensor = tf.Variable(1.0, name='m')
b_tensor = tf.Variable(1.0, name='b')

# 计算出所有的错误
error = 0
for x,y in zip(x_data, y_label):
    y_hat = m_tensor*x + b_tensor
    error += (y-y_hat)**2

# 优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

# 训练
with tf.Session() as sess:
    writer = tf.summary.FileWriter(dir + './graphs', sess.graph)

    sess.run(init)

    print('error=', error.eval())
    print('开始训练:\n')
    training_steps = 1000
    for i in range(training_steps):
        sess.run(train)
        print('m_tensor=',m_tensor.eval(), 'b_tensor=', b_tensor.eval(), 'error=', error.eval())
    
    final_slope, final_intercept = sess.run([m_tensor,b_tensor])
writer.close()

print('\n')
print('final_slope=',final_slope,'final_intercept=', final_intercept)
# print(type(final_slope)) # nump.float32

# 训练1000次的结果：final_slope= 0.29992807 final_intercept= 0.6005392
# 找到了正确的值