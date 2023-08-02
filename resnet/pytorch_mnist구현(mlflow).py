import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transfroms
import os
import tempfile

import mlflow
import mlflow.pytorch
from mlflow import log_metric, log_param, log_artifacts

#experiment_id = mlflow.create_experiment("conv") # 실험 이름을 바꿀 수 있음

# 쿠다 활용 가능하면 GPU(쿠다) 사용
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777) # 실행할 때마다 결과가 달라지지 않도록 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
print(device + " is available")

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
args = Params(100, 100, 5, 0.001, 0.5, 777, True, 200)

learning_rate = 0.001
batch_size = 100
num_classes = 10
epochs = 5

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
 
class ConvNet(nn.Module):
  def __init__(self): # layer 정의
        super(ConvNet, self).__init__()

        # input size = 28x28 
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # input channel = 1, filter = 10, kernel size = 5, zero padding = 0, stribe = 1
        # ((W-K+2P)/S)+1 공식으로 인해 ((28-5+0)/1)+1=24 -> 24x24로 변환
        # maxpooling하면 12x12
  
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # input channel = 1, filter = 10, kernel size = 5, zero padding = 0, stribe = 1
        # ((12-5+0)/1)+1=8 -> 8x8로 변환
        # maxpooling하면 4x4

        self.drop2D = nn.Dropout2d(p=0.25, inplace=False) # 랜덤하게 뉴런을 종료해서 학습을 방해해 학습이 학습용 데이터에 치우치는 현상을 막기 위해 사용
        self.mp = nn.MaxPool2d(2)  # 오버피팅을 방지하고, 연산에 들어가는 자원을 줄이기 위해 maxpolling
        self.fc1 = nn.Linear(320,100) # 4x4x20 vector로 flat한 것을 100개의 출력으로 변경
        self.fc2 = nn.Linear(100,10) # 100개의 출력을 10개의 출력으로 변경

  def forward(self, x):
        x = F.relu(self.mp(self.conv1(x))) # convolution layer 1번에 relu를 씌우고 maxpool, 결과값은 12x12x10
        x = F.relu(self.mp(self.conv2(x))) # convolution layer 2번에 relu를 씌우고 maxpool, 결과값은 4x4x20
        x = self.drop2D(x)
        x = x.view(x.size(0), -1) # flat
        x = self.fc1(x) # fc1 레이어에 삽입
        x = self.fc2(x) # fc2 레이어에 삽입
        return F.log_softmax(x) # fully-connected layer에 넣고 logsoftmax 적용
 
model = ConvNet().to(device) # CNN instance 생성
# Cost Function과 Optimizer 선택
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

def train(epoch):
    model.train()
    for epoch in range(epochs): # epochs수만큼 반복
        avg_cost = 0

        for batch_idx, (data, target) in enumerate(train_loader):
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
                #model.log_weights(step)
        #print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

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
            test_cost += F.nll_loss(out, target, reduction='sum').data.item() # sum up batch loss
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



with mlflow.start_run() as run:
    # mlflow ui에 파라미터 기록(tracking)
    # param이라고 하면 매개변수 탭에 저장, metric이라 하면 측정 항목에 저장, artifacts에는 로그 파일 같은거
    # log_param('learning_rate', learning_rate)
    # log_param('batch_size', batch_size)
    # log_param('num_classes', num_classes)
    # log_param('epochs', epochs)
    
    # Log our parameters into mlflow
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
        mlflow.log_artifacts(output_dir, artifact_path="events")
        #mlflow.log_metric(key="quality", value=2*epoch, step=epoch)

    # os 경로에 파일이 없으면 생성하고 특정 문자열을 그 파일에 쓴다 -> mlflow ui 실행시 artifacts 탭에 문자들이 나옴
    # path = "outputs"

    # if not os.path.exists(path):
    #     os.makedirs(path)

    # with open("%s/test.txt" %(path), 'w') as f:
    #     f.write("이 모델은 mnist 손글씨 데이터셋을 활용한 pytorch 모델입니다")

    # log_artifacts("%s" %(path))

    # 다음 코드를 실행하면 conv model 폴더가 생성되며 해당 디렉토리는 packages와 비슷한 파일을 포함하고 있으며 학습 결과인 model파일을 가지고 있음
    # mlflow.pytorch.log_model(model, "conv_model")
    #model_uri = mlflow.get_artifact_uri("conv_model")