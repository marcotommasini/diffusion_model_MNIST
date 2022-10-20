import import_ipynb       #This is a package that allows me to get functions directly from colab notebooks

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/Diffusion_model_training/MNIST_model
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.datasets.mnist import load_data
from skimage.transform import resize
from torch import nn
import torch
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x1 = F.relu(F.max_pool2d(self.conv1(x), 2))
        x1 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x1)), 2))
        x_flatten = x1.view(-1, 500)
        x2 = F.relu(self.fc1(x_flatten))
        x3 = F.dropout(x2, training=self.training)
        x_out = self.fc2(x3)
        return F.log_softmax(x_out), x2
        
        
class FID():
  def __init__(self, device, checkpoint_directory):

    self.device = device
    self.checkpoint_directory = checkpoint_directory

  # calculate frechet inception distance
  def calculate_fid(self,images1, images2):
    
    model = Net()
    model = model.to(self.device)
    model_checkpoint = torch.load(self.checkpoint_directory)
    model.load_state_dict(model_checkpoint['model_state_dict'])

    # calculate activations
    _,act1 = model(images1)
    _,act2 = model(images2)

    act1 = act1.cpu().detach().numpy()
    act2 = act2.cpu().detach().numpy()


    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
  
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
      covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid