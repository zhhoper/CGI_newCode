import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class MSELoss(nn.Module):
    def __init__(self):
    	super(MSELoss, self).__init__()
    def forward(self, groundTruth, output):
    	# compute MSE loss
    	# groundTruth and output are tensors with same size:
    	# NxCxHxW
    
    	if groundTruth.size()[0] != output.size()[0] or \
    		groundTruth.size()[1] != output.size()[1] or \
    		groundTruth.size()[2] != output.size()[2] or \
    		groundTruth.size()[3] != output.size()[3]:
    			print(groundTruth.size())
    			print(output.size())
    			raise ValueError('size of groud truth and ouptut does not match')
    
    	# numData = output.size()[0]
    	# total_loss = Variable(torch.cuda.FloatTensor([0]))
    
    	#for i in range(numData):
    	#	tmp_1 = groundTruth[i]
    	#	tmp_2 = output[i]
    	#	tmp = (tmp_1 - tmp_2)**2
    	#	loss = torch.sum(tmp)/(output.size()[1]*output.size()[2]*output.size()[3])
    	#	total_loss += loss
    	# return total_loss/numData
        return torch.sum((groundTruth - output)**2) \
                /(output.size()[0]*output.size()[1]*output.size()[2]*output.size()[3])

class mask_loss(nn.Module):
    '''
    	used for the first order term in augmented Lagrangian method
    '''
    def __init__(self):
    		super(mask_loss, self).__init__()
    def forward(self, mask, output):
    	# compute MSE loss
    	# groundTruth and output are tensors with same size:
    	# NxCxHxW
    
        if mask.size()[1] != output.size()[1] or \
        	mask.size()[2] != output.size()[2] or \
        	mask.size()[3] != output.size()[3]:
        		print(mask.size())
        		print(output.size())
        		raise ValueError('size of groud truth and ouptut does not match')
    
    	#numData = output.size()[0]
    	#total_loss = Variable(torch.cuda.FloatTensor([0]))
    	#channel = output.size()[1]
    	#for i in range(numData):
    	#	tmpOutput = output[i]
    	#	tmpMask = mask[i]
    	#	tmp = tmpMask*tmpOutput
    	#	#if np.isnan(tmp.sum().data.cpu().numpy()):
    	#	#		print 'tmp has nan'
    	#	#		raise
    	#	#if np.isnan(tmpMask.sum().data.cpu().numpy()):
    	#	#		print 'mask has none'
    	#	#		raise
    	#	loss = tmp.sum()/(tmpMask.sum() + 1e-6)
    	#	total_loss += loss
        #return total_loss/numData
		
        total_loss = (mask * output).sum()/(mask.sum() + 1e-6)
        return total_loss

class SiMSELoss(nn.Module):
	def __init__(self):
		super(SiMSELoss, self).__init__()
		self.eps = 1e-10
	def forward(self, groundTruth, output):
		# compute SiMSELoss according to the paper
		# get an alpha to minimize 
		#     (groundTruth - alpha*output)^2
		# then the loss is defined as the above 
		
		if groundTruth.size()[0] != output.size()[0] or \
			groundTruth.size()[1] != output.size()[1] or \
			groundTruth.size()[2] != output.size()[2] or \
			groundTruth.size()[3] != output.size()[3]:
				print(groundTruth.size())
				print(output.size())
				raise ValueError('size of groud truth and ouptut does not match')

		numData = output.size()[0]
		alpha = torch.sum(groundTruth*output)/(torch.sum(output**2) + self.eps)
		return torch.sum((groundTruth - alpha*output)**2) \
                /(output.size()[1]*output.size()[2]*output.size()[3]*numData)


class getLoss(object):
    def __init__(self, criterion, gradientLayer):
        super(getLoss, self).__init__()
        self.criterion = criterion
        self.gradientLayer = gradientLayer
        self.eps = 1e-6

    def getAlpha(self, source, target):
        '''
            compute a alpha to minimize
            (alpha*source - target)^2 
        '''
        if source.size()[0] != target.size()[0] or \
        	source.size()[1] != target.size()[1] or \
        	source.size()[2] != target.size()[2] or \
        	source.size()[3] != target.size()[3]:
                raise ValueError('size of groud truth and ouptut does not match')
        numImages = source.shape[0]
        source = source.contiguous()
        target = target.contiguous()
        source = source.view(numImages, -1)
        target = target.view(numImages, -1)
                    
        alpha = torch.sum(target*source,dim=1)/(torch.sum(source**2, dim=1) + self.eps)
        return alpha

    def getAlphaImage(self, source, target):
        '''
            output source = alpha*source so that source is close to target
        '''
        alpha = self.getAlpha(source, target)
        numImages, numChannels, numRows, numCols = source.shape
        tmp_alpha = alpha.view(numImages,1).repeat(1, numChannels, numRows, numCols)
        tmp_alpha_1 = Variable(tmp_alpha.view(numImages, numChannels, numRows, numCols).data.cuda()).float()
        return tmp_alpha_1*source
    
    def getLoss(self, output_albedo, output_shading, output_normal, albedo, shading, normal, mask, image):
        '''
            compute the loss for the mini-batch
            including reconstruction loss
        '''
        image = F.relu(image) + 1e-6 
        #alpha_albedo = self.getAlphaImage(output_albedo, albedo)
        #alpha_shading = self.getAlphaImage(output_shading, shading)
        alpha_albedo = output_albedo
        alpha_shading = output_shading
        alpha_image = output_albedo*output_shading
    
        alpha_albedo_grad = self.gradientLayer.forward(alpha_albedo)
        albedo_grad = self.gradientLayer.forward(albedo)
        alpha_shading_grad = self.gradientLayer.forward(alpha_shading)
        shading_grad = self.gradientLayer.forward(shading)
        output_normal_grad = self.gradientLayer.forward(output_normal)
        normal_grad = self.gradientLayer.forward(normal)
        
        imgMask = mask.expand(-1, 3, -1, -1)
        gradMask = mask.expand(-1, 6, -1, -1)
        
        loss_albedo = self.criterion(imgMask, torch.abs(albedo - alpha_albedo))
        loss_shading = self.criterion(imgMask, torch.abs(shading - alpha_shading))
        #loss_albedo = self.criterion(imgMask, (albedo - alpha_albedo)**2)
        #loss_shading = self.criterion(imgMask, (shading - alpha_shading)**2)
        
        loss_albedo_grad = self.criterion(gradMask, torch.abs(albedo_grad - alpha_albedo_grad))
        loss_shading_grad = self.criterion(gradMask, torch.abs(shading_grad - alpha_shading_grad))
        
        loss_normal = self.criterion(mask, -1*(torch.sum(output_normal*normal, dim=1, keepdim=True)))
        loss_normal_grad = self.criterion(gradMask, torch.abs(normal_grad - output_normal_grad))
    
        loss_image = self.criterion(imgMask, (alpha_image - image**2.2)**2)
        #loss_image = self.criterion(imgMask, (alpha_image - image**2.2)**2)

        # get log loss
        log_alpha_albedo = torch.log(F.relu(alpha_albedo) + 1e-6)
        log_alpha_shading = torch.log(F.relu(alpha_shading) + 1e-6)
        log_output_normal = torch.log(F.relu(output_normal + 1) + 1e-6)
        log_alpha_image = torch.log(F.relu(alpha_image) + 1e-6)

        log_albedo = torch.log(F.relu(albedo) + 1e-6)
        log_shading = torch.log(F.relu(shading) + 1e-6)
        log_normal = torch.log(F.relu(normal + 1) + 1e-6)
        log_image = torch.log(F.relu(image**2.2) + 1e-6)

        log_loss = {}
        log_loss['albedo'] = self.criterion(imgMask, (log_albedo - log_alpha_albedo)**2)
        log_loss['shading'] = self.criterion(imgMask, (log_shading - log_alpha_shading)**2)
        log_loss['normal'] = self.criterion(imgMask, torch.abs(log_output_normal - log_normal))
        log_loss['image'] = self.criterion(imgMask, (log_alpha_image - log_image)**2)
        
        return loss_albedo, loss_shading, loss_albedo_grad, loss_shading_grad, loss_normal, loss_normal_grad, loss_image, log_loss
