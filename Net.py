import torch.nn as nn
import torch.nn.functional as F 

class testNet(nn.Module):
	def __init__(self):
		super(testNet,self).__init__()
		self.conv0 = nn.Conv2d(in_channels=1,out_channels =10,kernel_size = [1,1])
		self.conv1 = nn.Conv2d(in_channels=10,out_channels=20,kernel_size = [1,1])
		self.bn1 = nn.BatchNorm2d(10)
		self.bn2 = nn.BatchNorm2d(20)
		self.conv2 = nn.Conv2d(in_channels = 20,out_channels = 40,kernel_size= [1,1])
		self.conv3 = nn.Conv2d(in_channels = 40,out_channels = 20,kernel_size = [1,1])
		self.conv4 = nn.Conv2d(in_channels = 20,out_channels = 20,kernel_size = [1,1])

		self.fc0 = nn.Linear(320,120)
		self.fc1 = nn.Linear(120,60)
		self.fc2 = nn.Linear(60,4)
	def forward(self,x):
		x = F.relu(self.bn1(self.conv0(x)))
		x = F.relu(self.bn2(self.conv1(x)))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = x.view(-1,320)
		x = F.relu(self.fc0(x))
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x)

class myNet(nn.Module):
	def __init__(self):
		super(myNet,self).__init__()

		self.conv0=nn.Conv2d(1,32,kernel_size=1)
		self.conv1=nn.Conv2d(32,64,kernel_size=(1,2))
		self.conv2=nn.Conv2d(64,128,kernel_size=(2,1))
		self.conv3=nn.Conv2d(128,256,kernel_size=(1,2))
		self.conv4=nn.Conv2d(256,256,kernel_size=1)
		self.conv5=nn.Conv2d(256,128,kernel_size=(2,1))
		self.conv6=nn.Conv2d(128,64,kernel_size=(1,2))
		self.conv7=nn.Conv2d(64,32,kernel_size=(2,1))
		self.conv8=nn.Conv2d(32,32,kernel_size=1)

		self.fc1 = nn.Linear(32,64)
		self.fc2 = nn.Linear(64,16)
		self.fc3 = nn.Linear(16,4)
		self.drop= nn.Dropout2d(p=0.2)

	def forward(self, x):

		x=F.relu(self.conv0(x))
		x=F.relu(self.conv1(x))
		x=F.relu(self.conv2(x))
		x=F.relu(self.conv3(x))
		x=self.drop(x)

		x=F.relu(self.conv4(x))
		x=F.relu(self.conv5(x))
		x=F.relu(self.conv6(x))
		x=F.relu(self.conv7(x))
		x=F.relu(self.conv8(x))
		
		x=x.view(-1,32)
		x=self.drop(x)

		x=F.relu(self.fc1(x))
		x=F.relu(self.fc2(x))
		x=self.fc3(x)
		return F.log_softmax(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 1) 
        self.conv2 = nn.Conv2d(32, 64, 2)  
        self.fc1   = nn.Linear(32*1*2, 120)  
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x): 
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) 
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) 
        x = x.view(x.size()[0], -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)        
        return x


		
