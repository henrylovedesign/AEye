import set_caffe_path
import tempfile

from caffe import layers as L, params as P
import caffe
import sys
from caffe.proto import caffe_pb2


caffe.set_mode_gpu();
caffe.set_device(0);

kernel_size={
	"conv1":4,
	"pool1":3,
	"conv2":4,
	"pool2":3
	}
stride={
	"pool1":2,
	"pool2":2
}
pool={
	"pool1":P.Pooling.MAX,
	"pool2":P.Pooling.MAX,
}
num_output={
	"conv1":64,
	"conv2":32,
	"fc1":200,
	"score":2
}
weight_filler={
	"conv1":dict(type='xavier'),
	"conv2":dict(type='xavier'),
	"relu1":dict(type='xavier'),
	"fc1":dict(type='xavier'),
	"score":dict(type='xavier')
}


def lenet(imagedata,batch_size,mode="TRAIN"):
	n=caffe.NetSpec()
	if mode=="TRAIN":
		shuffle=True

	else:
		shuffle=False
	n.data,n.label=L.ImageData(batch_size=batch_size, source=imagedata,
                             transform_param=dict(scale=1./255), shuffle=shuffle,ntop=2)
	
	

	n.conv1=L.Convolution(n.data,kernel_size=kernel_size["conv1"],num_output=num_output["conv1"],weight_filler=weight_filler["conv1"])
	n.pool1=L.Pooling(n.conv1,kernel_size=kernel_size["pool1"],stride=stride["pool1"],pool=pool["pool1"])
	n.relu1=L.ReLU(n.pool1, in_place=True)
	n.conv2=L.Convolution(n.relu1,kernel_size=kernel_size["conv2"],num_output=num_output["conv2"],weight_filler=weight_filler["conv2"])
	n.pool2=L.Pooling(n.conv2,kernel_size=kernel_size["pool2"],stride=stride["pool2"],pool=pool["pool2"])
	n.relu2=L.ReLU(n.pool2, in_place=True)
	n.fc1=L.InnerProduct(n.relu2,num_output=num_output["fc1"],weight_filler=weight_filler["fc1"])
	n.relu3=L.ReLU(n.fc1,in_place=True)
	n.score=L.InnerProduct(n.relu3,num_output=num_output["score"],weight_filler=weight_filler["score"])
	n.loss=L.SoftmaxWithLoss(n.score, n.label)


	return n.to_proto()

def creat_netproto(protonet,protofile):
	with open(protofile, 'w') as f:
		f.write(str(protonet))

def solver(train_net_path,solverproto,test_net_path=None,base_lr=0.001):
	s=caffe_pb2.SolverParameter()

	
	s.train_net = train_net_path
	if test_net_path is not None:
		s.test_net.append(test_net_path)

		s.test_iter.append(2)

		s.test_interval = 20


	s.iter_size=7

	s.max_iter = 240

	s.type = 'SGD'

	s.base_lr = base_lr

	s.lr_policy='step'
	s.gamma=0.1
	s.stepsize=100


	s.momentum = 0.9
	s.weight_decay = 0.0005


	s.display = 10

	s.solver_mode = caffe_pb2.SolverParameter.GPU
	s.snapshot=240
	s.snapshot_prefix="/home/henry/projects/toBeLable/R2D2/"

	with open(solverproto,'w') as f:
		f.write(str(s))
		print f.name
	f.close()
	return f.name


traindata="/home/henry/projects/toBeLable/R2D2/train.data"
testdata="/home/henry/projects/toBeLable/R2D2/test.data"
batch_size_train=800
batch_size_test=40

protonet_train = lenet(traindata,batch_size_train)
protonet_test = lenet(testdata,batch_size_test,"TEST")

protofile_train="/home/henry/projects/toBeLable/R2D2/train.prototxt"
protofile_test="/home/henry/projects/toBeLable/R2D2/test.prototxt"

creat_netproto(protonet_train,protofile_train)
creat_netproto(protonet_test,protofile_test)
train_net_path=protofile_train
test_net_path=protofile_test

solverproto="/home/henry/projects/toBeLable/R2D2/solver.prototxt"
solverfile = solver(train_net_path,solverproto,test_net_path, base_lr=0.001)
s=caffe.get_solver(solverfile)

niter=240

test_interval=20
test_iter=15
for it in range(niter):
	s.step(1)
	#s.test_nets[0].forward(start='conv1')

	if it%test_interval ==0:
		correct=0
		for test_it in range(test_iter):
			s.test_nets[0].forward()

			correct+=sum(s.test_nets[0].blobs['score'].data.argmax(1)
                           == s.test_nets[0].blobs['label'].data)
			print s.test_nets[0].blobs['score'].data.argmax(1)
			

			print s.test_nets[0].blobs['label'].data
			#print correct
		print float(correct) / 600
	
