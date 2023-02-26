import cv2
from keras import layers
import tensorflow as tf
import os
import json
from keras import Model
from matplotlib import pyplot as plt
from tensorflow.python.keras.utils import np_utils
import numpy as np
from tensorflow.keras.optimizers import SGD
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
train_path = 'F:/深度学习课程设计/train_dataset1/train'#训练集路径
test_path = 'F:/深度学习所需：/test'             #测试集路径
train_y = []  #保存数据 x代表数据，y代表类别
train_x  = []
test_x = []
test_y = []
js = {"annotation":[]}
index = 0
for i in os.listdir(train_path):  #遍历训练集路径下所有的文件返回list列表形式
    file_path = train_path + '//' + i #生成文件路径
    for w in os.listdir(file_path): #遍历文件路径下所有的图片返回list列表形式
        img_path = file_path + '//' + w#生成图像路径
        img = plt.imread(img_path)    #读取图片
        # print(img)
        image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#img转opencv
        train_x.append(image) #图像增强技术的作用
        train_y.append(index)
    index = index + 1
print('训练集样本已生成.......')
index = 0
for i in os.listdir(test_path):#遍历测试集路径下所有的文件返回list列表形式
    file_path = test_path + '//' + i#生成文件路径
    for w in os.listdir(file_path): #遍历文件路径下所有的图片返回list列表形式
        filename = 'test/'+i+'/'+w #生成图像路径
        j = {}
        j['filename'] = filename
        j['label'] = 0
        js['annotation'].append(j)
        img_path = file_path + '//' + w
        img = plt.imread(img_path)
        image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#img转opencv
        test_x.append(image)
        test_y.append(index)
    index = index + 1
print('测试集样本已生成.......')
train_x = np.array(train_x,dtype = 'float32')
train_y = np.array(train_y,dtype = 'float32')
test_x = np.array(test_x,dtype = 'float32')
test_y = np.array(test_y,dtype = 'float32')

print('数据添加一维...')
train_x  = train_x.reshape(6024, 224, 224,1)#读取训练集数据，改变数据集格式  共6024张图片，每张图片是224*224
test_x = test_x.reshape(324, 224, 224,1)   #读取测试集数据，改变数据集格式  共324张图片，每张图片是224*224

train_x = np.array(train_x,dtype = 'float32')#将数据转化为浮点数类型
train_y = np.array(train_y,dtype = 'float32')
test_x = np.array(test_x,dtype = 'float32')
test_y = np.array(test_y,dtype = 'float32')
#归一化
train_x /= 255
test_x /= 255
# 对标签进行one_hot编码
print('one_hot编码........')
y_train_new = np_utils.to_categorical(num_classes = 10,y = train_y)#实现one-hot编码，num_classes=10表示10个类别
y_test_new  = np_utils.to_categorical(num_classes = 10,y = test_y)


def VI (class_nums):
    '''主模型'''
    inpt = layers.Input(shape=(224,224,1))#第一层输入层，指定长宽均为224，通道数为1
    cv1 = layers.Conv2D(filters=32, kernel_size=(5,5),activation='relu')(inpt)#第一个卷积层，32个卷积核，大小5*5，激活函数relu
    ma1 = layers.MaxPool2D(pool_size=(2,2))(cv1)#第一个池化层，池化核大小2*2
    drop1 = layers.Dropout(0.25)(ma1)#随机丢弃四分之一的网络连接，防止过拟合
    cv2 = layers.Conv2D(filters=64, kernel_size=(3, 3),activation='relu')(drop1)#第二个卷积层，64个卷积核，大小3*3，激活函数relu
    ma2 = layers.MaxPool2D(pool_size=(2, 2))(cv2)#第二个池化层，池化核大小为2*2
    drop2 = layers.Dropout(0.25)(ma2)#随机丢弃四分之一的网络连接，防止过拟合
    flat = layers.Flatten()(drop2)#全连接层展开操作
    drop2 = layers.Dropout(0.5)(flat)#随机丢弃二分之一的网络连接，防止过拟合
    dence2 = layers.Dense(class_nums, activation='softmax')(drop2)#输出层
    model = tf.keras.Model(inputs=inpt, outputs=dence2)#从inpt这个层输入，从dence2这个层输出
    return model

#开始训练
class_num = 10
print('模型开始加载.......')
model = VI(class_num)
# model.summary()
model.compile( optimizer='adam',                # 选择adam优化器
                loss="categorical_crossentropy", # 使用交叉熵损失函数
                metrics=["acc"]                  # 使用acc评估模型的结果
                    )

print('模型开始训练.......')               #32          10
model.fit(train_x, y_train_new, batch_size=32, epochs=10, validation_split=0.2) #单次传递给程序用以训练的样本数据个数为32，epoch指的是被前向传递和后向传递到神经网络一次；validation_split=0.2表示从训练集中抽取20%的作为验证集
print('模型训练完成，开始预测.......')
#验证结果
pred_y = model.predict(test_x)#预测值

print('模型预测集生成，保存模型.......')
model.save('jtmodel_last_model.h5')#保存加载模型
np.save('jtmodel_last_model.npy',pred_y)
eval = model.evaluate(test_x,y_test_new)#对模型进行评估
print('损失值:',eval[0],'准确率:',eval[1])

print('开始生成json文件.......')

for i in range(324):
    js['annotation'][i]['label'] = list(pred_y[i]).index(max(list(pred_y[i])))#将图片放入模型预测，返回一个列表，列表为该条数据属于十种类别的概率，找到最大概率的下标，就是这个图片预测出来的图片
# print(js)
js=json.dumps(js)#将dict数据转化为str
with open('submit.json','w+') as file:#将结果写入到submit这个文件夹里面
    file.write(js)

print('训练过程可视化生成.......')

p = []
for i in range(len(pred_y)):
    p.append(list(pred_y[i]).index(max(list(pred_y[i]))))#将其转化成类别的数据集
#真实标签与预测标签对比可视化
plt.plot(test_y,label='Truthlabel')
plt.ylim(-1,10)
plt.ylabel('labels')
plt.xlabel('photos')
plt.plot(p,label='Predictlabel')
plt.legend()
plt.title('真实标签和预测标签对比')
plt.show()
#训练过程损失值可视化
loss = [1.5399,0.8733,0.5905,0.5037,0.4736,0.4450,0.4214,0.3798,0.3692,0.3690]
val_loss = [1.3867,0.6276,0.5581,0.5077,0.4725,0.4992,0.4635,0.4609,0.4599,0.4376]
acc = [0.5343,0.7425,0.8373,0.8593,0.8655,0.8736,0.8790,0.8867,0.8965,0.8900]
val_acc = [0.5145,0.8423,0.8382,0.8647,0.8714,0.8581,0.8631,0.8705,0.8747,0.8697]

plt.plot(range(1,11),loss,label = 'loss')#训练过程中的损失率
plt.plot(range(1,11),val_loss,label = 'val_loss')#在训练集中拿出一部分测试所得的测试损失率

# plt.ylim(0,11)
plt.ylabel('损失值')
plt.xlabel('epoch')
plt.legend()
plt.title('训练过程损失值可视化')
plt.show()
#训练过程准确率可视化
plt.plot(range(1,11),acc,label = 'acc')#acc表示训练过程中的正确率
plt.plot(range(1,11),val_acc,label = 'val_acc')#val_acc表示在训练集中拿出一部分测试所得的测试准确率

# plt.ylim(0,11)
plt.ylabel('准确率')
plt.xlabel('epoch')
plt.legend()
plt.title('训练过程准确率可视化')
plt.show()