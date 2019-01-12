import torch
import torch.nn.functional as F 
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import os
import pandas as pd
import numpy as np
from Net import myNet
from Net import testNet
from Net import Net
batchSize=32

#数据预处理
class dataset(torch.utils.data.Dataset):

	def __init__(self, file, transform=None, targetTransform=None):
		dataframe = pd.read_csv(file)
		dataArray = dataframe.values

		self.data = dataArray[:, 0:16]
		self.label = dataArray[:, 16]
		self.transform = transform
		self.targetTransform = targetTransform

	def __getitem__(self, index):
		board = self.data[index].reshape((4, 4))
		board = board[:, :, np.newaxis]
		
		label = self.label[index]

		if self.transform is not None:
			board = self.transform(board)
		return board, label

	def __len__(self):
		return len(self.label)

#加载数据
#device = torch.device('cuda') if torch.cuda.available() else torch.device('cpu')
def loadData():
	#训练集
	trainingData = dataset(file= './training/training.csv',		 
		transform=transforms.Compose([transforms.ToTensor()]))

	trainloader = torch.utils.data.DataLoader(
		trainingData,batch_size=batchSize,shuffle=True,num_workers=2)
	#测试集
	testData = dataset(file= './training/test.csv',
        	transform=transforms.Compose([transforms.ToTensor()]))

	testloader = torch.utils.data.DataLoader(
		testData,batch_size=batchSize,shuffle=True,num_workers=2)

	return trainloader, testloader

#训练网络
def train():
	net=myNet()
	optimizer=optim.Adam(net.parameters(),lr=0.0001)
	trainloader, testloader = loadData()
	epochs=100
	for epoch in range(epochs):	
		running_loss=0.0
		for i, (data,target) in enumerate(trainloader,0):
			data = data.type(torch.float)

			if torch.cuda.is_available():
				data=Variable(data).cuda()
				target=Variable(target).cuda()
				net.cuda()
			#梯度清零
			optimizer.zero_grad()
			#forward + backward
			outputs = net(data)
			loss=F.nll_loss(outputs,target)
			loss.backward()
			#更新参数
			optimizer.step()
			
			#打印log信息
			running_loss += loss.item()
			if i%100==99:
				print('epoch: %d loss:%.3f '%(epoch+1,running_loss/100))
				running_loss = 0.0
				#预测
				predict = outputs.data.max(1)[1]
				number = predict.eq(target.data).sum()
				correct = 100*number/batchSize
				#print("\t",predict[0:20])
				#print("\t",target[0:20])
				print('Accuracy:%0.2f'%correct,'%')
		torch.save(net.state_dict(),'model.pth' )

if __name__=='__main__':
	train()
