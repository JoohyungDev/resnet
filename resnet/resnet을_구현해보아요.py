import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transfroms
import os
import tempfile
from torchvision import models
import numpy as np

import mlflow
import mlflow.pytorch
from mlflow import log_metric, log_param, log_artifacts

from mlflow.models.signature import infer_signature

#resnet18_pretrained = models.resnet18(pretrained=True)

# 사전훈련된 resnet 불러와서 수정하기
class resnet18_pretrained(nn.Module): # MnistResNet은 nn.Module 상속
  def __init__(self, in_channels=1):
    super(resnet18_pretrained, self).__init__()

    # torchvision.models에서 사전훈련된 resnet 모델 가져오기
    self.model = models.resnet18(pretrained=True)

    # 기본 채널이 3(RGB)이기 때문에 mnist에 맞게 1(grayscale image)로 바꿔준다.  
    # 원래 ResNet의 첫번째 층
    # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # 1000개 클래스 대신 10개 클래스로 바꿔주기
    num_ftrs = self.model.fc.in_features
    # nn.Linear(in_features, out_features ...)
    self.model.fc = nn.Linear(num_ftrs, 10)
    
  def forward(self, x): # 모델에 있는 foward 함수 그대로 가져오기
    return self.model(x)


resnet18_pretrained = resnet18_pretrained()

# change the output layer to 10 classes
# num_classes = 10
# num_ftrs = resnet18_pretrained.fc.in_features
# resnet18_pretrained.fc = nn.Linear(num_ftrs, num_classes)

device = torch.device('cuda:0')
resnet18_pretrained.to(device)

# # get the model summary
# from torchsummary import summary
# summary(resnet18_pretrained, input_size=(1, 28, 28), device=device.type)


# Create Params dictionary
class Params(object):
    def __init__(self, batch_size, test_batch_size, epochs, lr, momentum, seed, cuda, log_interval):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.seed = seed
        self.cuda = cuda
        self.log_interval = log_interval

# Configure args
args = Params(64, 64, 1, 0.001, 0.5, 777, True, 200)

learning_rate = 0.001
batch_size = 64
num_classes = 10
epochs = 1  # 원래는 5였음

# MNIST 데이터셋 로드
train_set = torchvision.datasets.MNIST(
    root = '/home/pjh/mnist_data',
    train = True,
    download = True,
    transform = transfroms.Compose([
        transfroms.ToTensor() # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    ])
)

test_set = torchvision.datasets.MNIST(
    root = '/home/pjh/mnist_data',
    train = False,
    download = True,
    transform = transfroms.Compose([
        transfroms.ToTensor() # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    ])
)




# train_loader, test_loader 생성(실제로 학습할 때 이용할 수 있게 배치사이트 형태로 만들어주는 라이브러리)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)


# input size를 알기 위해서
examples = enumerate(train_set)
batch_idx, (example_data, example_targets) = next(examples)
example_data.shape

model = resnet18_pretrained.to(device) # resnet18 instance 생성

# Cost Function과 Optimizer 선택
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)  

def train(epoch):
    model.train()
    for epoch in range(epochs): # epochs수만큼 반복
        avg_cost = 0

        for batch_idx, (data, target) in enumerate(train_loader): # batch_idx = batch의 index
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad() # 모든 model의 gradient 값을 0으로 설정
            hypothesis = model(data) # 모델을 forward pass해 결과값 저장
            cost = criterion(hypothesis, target) # output과 target의 loss 계산.
            cost.backward() # backward 함수를 호출해 gradient 계산
            optimizer.step() # 모델의 학습 파라미터 갱신
            avg_cost += cost / len(train_loader) # loss 값을 변수에 누적하고 train_loader의 개수로 나눔 = 평균
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), cost.data.item()))
                step = epoch * len(train_loader) + batch_idx
                log_scalar('train_loss', cost.data.item(), step)

def test(epoch):
# test
    test_cost = 0
    model.eval() # evaluate mode로 전환 dropout 이나 batch_normalization 해제 
    with torch.no_grad(): # grad 해제 
        correct = 0
        total = 0

        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            out = model(data)
            #test_cost += F.nll_loss(out, target, reduction='sum').data.item() # sum up batch loss
            cost = criterion(out, target)
            test_cost += cost.item()
            preds = torch.max(out.data, 1)[1] # 출력이 분류 각각에 대한 값으로 나타나기 때문에, 가장 높은 값을 갖는 인덱스를 추출
            total += len(target) # 전체 클래스 개수 
            correct += (preds==target).sum().item() # 예측값과 실제값이 같 은지 비교

    test_cost /= len(test_loader.dataset)
    test_accuracy = 100.0 * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_cost, correct, len(test_loader.dataset), test_accuracy))
    step = (epoch + 1) * len(train_loader)
    log_scalar('test_loss', test_cost, step)
    log_scalar('test_accuracy', test_accuracy, step)

def log_scalar(name, value, step):
    """Log a scalar value to MLflow"""
    mlflow.log_metric(name, value, step=step)

with mlflow.start_run() as run:         # mlflow.start_run creates a new MLflow run to track the performance of this model. 
    for key, value in vars(args).items():
        mlflow.log_param(key, value)
    
    output_dir = tempfile.mkdtemp()
    print("Writing Pytorch events locally to %s\n" % output_dir)

     # 매트릭 시각화(그래프)
    for epoch in range(1, args.epochs+1):
        # print out active_run
        print("Active Run ID: %s, Epoch: %s \n" % (run.info.run_uuid, epoch))

        train(epoch)
        test(epoch)

        print("Uploading Pytorch events as a run artifact.")
        mlflow.log_artifacts(output_dir, artifact_path="events") # mlflow 상에 artifact를 기록
        mlflow.pytorch.log_model(model,'model') # mlflow 상에 모델을 기록
    
    
print(train_loader)
print(test_loader)
#mlflow.pytorch.save_model(model, "resnet_model") # 모델 로컬에 저장(홈 경로)
# signature = infer_signature(input data)   # MLmodel 파일 부가 설명(입출력)

# Model Serving
import requests
import pandas as pd

def score_model(dataset: pd.DataFrame):
  url = 'https://<DATABRICKS_URL>/model/wine_quality/Production/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'}
  data_json = dataset.to_dict(orient='split')
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()