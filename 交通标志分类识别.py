# 解压数据集
!echo y | unzip test_dataset.zip > log.log
!echo y | unzip train_dataset.zip > log.log
import io, cv2
import math, json
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
import paddle
import paddle.nn.functional as F
import paddle.vision.transforms as T
from paddle.io import DataLoader, Dataset

import warnings
warnings.filterwarnings("ignore")

paddle.__version__
# 读取数据集
train_json = pd.read_json('train.json')
train_json = train_json.sample(frac=1.0)

train_json['filename'] = train_json['annotations'].apply(lambda x: x['filename'])
train_json['label'] = train_json['annotations'].apply(lambda x: x['label'])
train_json.head()
train_json['label'].value_counts()
plt.figure(figsize=(10, 10))
for idx in range(10):
    plt.subplot(1, 10, idx+1)
    img = cv2.imread(train_json['filename'].iloc[idx])
    plt.imshow(img)
    plt.xticks([]); plt.yticks([])


# 定义数据集
class WeatherDataset(Dataset):
    def __init__(self, df):
        super(WeatherDataset, self).__init__()
        self.df = df

        # 数据扩增方法
        self.transform = T.Compose([
            T.Resize(size=(128, 128)),
            T.RandomCrop(size=(125, 125)),
            T.RandomRotation(10),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5)
        ])

    def __getitem__(self, index):
        file_name = self.df['filename'].iloc[index]
        img = Image.open(file_name)
        img = self.transform(img)
        return img, paddle.to_tensor(self.df['label'].iloc[index])

    def __len__(self):
        return len(self.df)
# 训练集
train_dataset = WeatherDataset(train_json.iloc[:-500])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 验证集
val_dataset = WeatherDataset(train_json.iloc[-500:])
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
from paddle.vision.models import resnet18

# 定义模型
class WeatherModel(paddle.nn.Layer):
    def __init__(self):
        super(WeatherModel, self).__init__()
        backbone = resnet18(pretrained=True)
        backbone.fc = paddle.nn.Identity()
        self.backbone = backbone
        self.fc1 = paddle.nn.Linear(512, 10)

    def forward(self, x):
        out = self.backbone(x)
        logits1 = self.fc1(out)
        return logits1
model = WeatherModel()
model(paddle.to_tensor(np.random.rand(10, 3, 125, 125).astype(np.float32)))
# 优化器与损失函数
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.0001)
criterion = paddle.nn.CrossEntropyLoss()

# 模型训练与验证
for epoch in range(0, 4):
    Train_Loss, Val_Loss = [], []
    Train_ACC1 = []
    Val_ACC1 = []

    model.train()
    for i, (x, y1) in enumerate(train_loader):
        pred1 = model(x)
        loss = criterion(pred1, y1)
        Train_Loss.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        Train_ACC1.append((pred1.argmax(1) == y1.flatten()).numpy().mean())

    model.eval()
    for i, (x, y1) in enumerate(val_loader):
        pred1 = model(x)
        loss = criterion(pred1, y1)
        Val_Loss.append(loss.item())
        Val_ACC1.append((pred1.argmax(1) == y1.flatten()).numpy().mean())

    if epoch % 1 == 0:
        print(f'\nEpoch: {epoch}')
        print(f'Loss {np.mean(Train_Loss):3.5f}/{np.mean(Val_Loss):3.5f}')
        print(f'ACC {np.mean(Train_ACC1):3.5f}/{np.mean(Val_ACC1):3.5f}')
import glob
test_df = pd.DataFrame({'filename': glob.glob('./test/*/*.jpg')})
test_df['label'] = 0
test_df = test_df.sort_values(by='filename')
test_dataset = WeatherDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
model.eval()
pred = []

# 模型预测
for i, (x, y1) in enumerate(test_loader):
    pred1 = model(x)
    pred += pred1.argmax(1).numpy().tolist()

test_df['label'] = pred
test_df['label'].value_counts()
submit_json = {
    'annotations':[]
}

# 生成提交结果文件
for row in test_df.iterrows():
    submit_json['annotations'].append({
        'filename': 'test_images/' + row[1].filename.split('/')[-1],
        'label': row[1].label,
    })

with open('submit.json', 'w') as up:
    json.dump(submit_json, up)