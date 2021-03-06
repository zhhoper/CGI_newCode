'''
    get lightings for each pixel
'''

import sys
sys.path.append('utils')
sys.path.append('model')

import os
import numpy as np
import time

import torch
# set random seed for all possible randomness
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from defineHourglass_64 import *
from defineDRN_normal import *
from utils_shading import *
from defineCriterion import *
from loadData_CGI import *
from defineHelp import *
from defineHelp_lighting import *

# image size to deal with
IMAGE_SIZE=128

# data folder
#databaseFolder = '/net/acadia7a/data/xiangyu/hao_intrinsic/intrinsics_final'
#listFolder = './list'
listFolder = '/scratch1/data/CGI/'
databaseFolder = '/scratch1/data/CGI/'

criterion = mask_loss()

my_network_coarse = HourglassNet(27)
my_network_coarse.cuda()
my_network_fine = DRN(6, 6, 12)
my_network_fine.cuda()
my_network = DRN(6, 6, 12)
my_network.cuda()
gradientLayer = gradientLayer_color()
gradientLayer.cuda()
shadingLayer_coarse = constructShading()
shadingLayer_coarse.cuda()
shadingLayer = constructShading_lightImg()
shadingLayer.cuda()
samplingLightLayer = samplingLight()
samplingLightLayer.cuda()

getLoss = getLoss(criterion, gradientLayer)

transformer = transforms.Compose([cropImg(output_size=IMAGE_SIZE), ToTensor()])
transformer_test = transforms.Compose([testTransfer(output_size=IMAGE_SIZE), ToTensor()])

trainLoaderHelper = CGI(dataFolder = os.path.join(databaseFolder, 'images'), 
    albedoFolder=os.path.join(databaseFolder, 'images'),
    shadingFolder=os.path.join(databaseFolder, 'shading'),
	normalFolder=os.path.join(databaseFolder, 'normal'),
    maskFolder=os.path.join(databaseFolder, 'mask'),
    fileListName=os.path.join(listFolder, 'training.list'),
	missingListName=os.path.join(listFolder, 'missingData_v2.txt'),
    transform = transformer)

testLoaderHelper = CGI(dataFolder=os.path.join(databaseFolder, 'images'),
	albedoFolder=os.path.join(databaseFolder, 'images'),
	shadingFolder=os.path.join(databaseFolder, 'shading'),
	normalFolder=os.path.join(databaseFolder, 'normal'),
	maskFolder=os.path.join(databaseFolder, 'mask'),
	fileListName=os.path.join(listFolder, 'validation.list'),
	missingListName=os.path.join(listFolder, 'missingData_v2.txt'),
	transform = transformer_test)

trainLoader = torch.utils.data.DataLoader(trainLoaderHelper, 
    batch_size=5, shuffle=True, num_workers=5)

testLoader = torch.utils.data.DataLoader(testLoaderHelper, 
    batch_size=5, shuffle=False, num_workers=5)

def networkForward(data, log_weight, optimizer, Testing=False):
    '''
        given data, compute the loss for the network
        return loss in a list
    '''
    inputs, albedo, shading, normal, mask, _ = data
    inputs, albedo, shading, normal, mask = \
            Variable(inputs.cuda(), volatile=Testing).float(), \
            Variable(albedo.cuda(), volatile=Testing).float(), \
            Variable(shading.cuda(), volatile=Testing).float(), \
            Variable(normal.cuda(), volatile=Testing).float(), \
            Variable(mask.cuda(), volatile=Testing).float()

    # --------------------------------------------------------------------------------
    # get albedo, shading, normal, lighting in coarse scale and prepare the residual
    coarse_albedo, coarse_normal, coarse_lighting = \
            my_network_coarse(F.upsample(inputs, size=[64, 64], mode='bilinear'))
    coarse_shading = shadingLayer_coarse(coarse_normal, coarse_lighting)
    coarse_albedo = F.upsample(coarse_albedo, size=[128, 128], mode='bilinear')
    coarse_normal = F.upsample(coarse_normal, size=[128, 128], mode='bilinear')
    coarse_shading = F.upsample(coarse_shading, size=[128, 128], mode='bilinear')

    # NOTE: we have a bug in coarse network for lighting, correct it
    coarse_lighting = Variable(coarse_lighting[:,0:27].data).float()
    coarse_lighting = coarse_lighting.unsqueeze(-1)
    coarse_lighting = coarse_lighting.unsqueeze(-1)
    
    # concatenate images, albedo, normal, shading as input
    fine_inputs = F.upsample(inputs, size=[128, 128], mode='bilinear')
    inputs_albedo = torch.cat((fine_inputs, coarse_albedo), dim=1)
    inputs_normal = torch.cat((fine_inputs, coarse_normal), dim=1)
    inputs_lighting = torch.cat((fine_inputs, coarse_albedo, coarse_normal, coarse_shading), dim=1)
    # predict residual
    output_albedo, output_normal, output_lighting = \
            my_network_fine(inputs_albedo, inputs_normal, inputs_lighting)

    fine_lighting = output_lighting + coarse_lighting.expand(-1,-1, 128, 128)
    fine_albedo = output_albedo + coarse_albedo
    fine_normal = F.normalize(output_normal + coarse_normal, p=2, dim=1)
    fine_shading = shadingLayer(F.normalize(fine_normal, p=2,dim=1), fine_lighting)

    #-----------------------------------------------------------------------------------------
    # third stage
    #-----------------------------------------------------------------------------------------

    fine_albedo = F.upsample(fine_albedo, size=[IMAGE_SIZE, IMAGE_SIZE], mode='bilinear')
    fine_normal = F.upsample(fine_normal, size=[IMAGE_SIZE, IMAGE_SIZE], mode='bilinear')
    fine_shading = F.upsample(fine_shading, size=[IMAGE_SIZE, IMAGE_SIZE], mode='bilinear')
    fine_lighting = F.upsample(fine_lighting, size=[IMAGE_SIZE, IMAGE_SIZE], mode='bilinear')

    inputs_albedo = torch.cat((inputs, fine_albedo), dim=1)
    inputs_normal = torch.cat((inputs, fine_normal), dim=1)
    inputs_lighting = torch.cat((inputs, fine_albedo, fine_normal, fine_shading), dim=1)

    output_albedo, output_normal, output_lighting = my_network(inputs_albedo, inputs_normal, inputs_lighting)
    true_lighting = output_lighting + fine_lighting
    true_albedo = output_albedo + fine_albedo
    true_normal = F.normalize(output_normal + fine_normal, p=2, dim=1)
    true_shading = shadingLayer(F.normalize(true_normal, p=2,dim=1), true_lighting)

    loss_albedo, loss_shading, loss_albedo_grad, loss_shading_grad, \
        loss_normal, loss_normal_grad, loss_image, log_loss = getLoss.getLoss(true_albedo, 
        true_shading, true_normal, albedo, shading, normal, mask, inputs)

    lightMask = mask.expand(-1, 27, -1, -1)
    loss_light = criterion(lightMask, true_lighting**2)

    # append all the losses
    loss = {}
    loss['albedo'] = loss_albedo
    loss['albedo_grad'] = loss_albedo_grad
    loss['shading'] = loss_shading
    loss['shading_grad'] = loss_shading_grad
    loss['normal'] = loss_normal
    loss['normal_grad'] = loss_normal_grad
    loss['light'] = loss_light
    loss['loss_image'] = loss_image

    loss['log_albedo'] = log_loss['albedo']
    loss['log_shading'] = log_loss['shading']
    loss['log_normal'] = log_loss['normal']
    loss['log_image'] = log_loss['image']

    loss['loss_list'] = [loss_albedo.data[0], loss_albedo_grad.data[0], loss_shading.data[0],
        loss_shading_grad.data[0], loss_normal.data[0], loss_normal_grad.data[0], loss_light.data[0], 
        loss_image.data[0], log_loss['albedo'], log_loss['shading'], log_loss['normal'], log_loss['image']]

    # return loss
    total_loss = loss['albedo'] + loss['albedo_grad'] + \
            loss['shading'] + loss['shading_grad'] + \
            loss['normal'] + loss['normal_grad'] + \
            0.2*loss['light'] + loss['loss_image'] + \
            log_weight['albedo']*loss['log_albedo'] + \
            log_weight['shading']*loss['log_shading'] + \
            log_weight['normal']*loss['log_normal'] + \
            log_weight['image']*loss['log_image']

    if Testing == False:
        total_loss.backward()
        optimizer.step()

    return loss, true_albedo, true_normal, true_shading, output_normal, coarse_normal

def main(savePath, load_path_coarse, load_path_fine, log_weight, lr=1e-3, weight_decay=0, total_epoch=100):
    begin_time = time.time()
    print 'learning rate is %.6f' % lr
    print 'weight decay is %.6f' % weight_decay
    print 'epoch is %05d' % total_epoch
    savePath = savePath + '_{:0.6f}_{:0.6f}_{:06d}'.format(lr, weight_decay, total_epoch)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    saveIntermedia = os.path.join(savePath, 'trained_model')
    if not os.path.exists(saveIntermedia):
        os.makedirs(saveIntermedia)
    writer = SummaryWriter(os.path.join(savePath, 'tensorboard'))
    
    global my_network_coarse 
    my_network_coarse.load_state_dict(torch.load(load_path_coarse))
    my_network_coarse.cuda()
    my_network_coarse.train(False)

    my_network_fine.load_state_dict(torch.load(load_path_fine))
    my_network_fine.cuda()
    my_network_fine.train(False)

    monitor_count = 20
    optimizer = optim.Adam(my_network.parameters(), lr = lr, weight_decay=weight_decay)
    fid = open(os.path.join(savePath, 'training.log'), 'w')
    fid_sep = open(os.path.join(savePath, 'training_sep.log'), 'w')
    fid_test = open(os.path.join(savePath, 'testing.log'), 'w')
    fid_test_sep = open(os.path.join(savePath, 'testing_sep.log'), 'w')
    print>>fid_sep, 'albedo, albedo_grad, shading, shading_grad, normal, normal_grad, lighting, image,' + \
            'log_albedo, log_shading, log_normal, log_image'
    print>>fid_test_sep, 'albedo, albedo_grad, shading, shading_grad, normal, normal_grad, lighting, image, ' + \
            'log_albedo, log_shading, log_normal, log_image'

    numIte_training = 0
    numIte_testing = 0
    for epoch in range(total_epoch):
        # ---------------------------------------------------------
        # training
        # ---------------------------------------------------------
        running_loss = 0.0
        # loss for each indivisual component
        running_albedo_loss = 0.0
        running_shading_loss = 0.0
        running_normal_loss = 0.0
        running_light_loss = 0.0
        running_image_loss = 0.0
        running_log_albedo = 0.0
        running_log_shading = 0.0
        running_log_image = 0.0
        running_log_normal = 0.0
        
        my_network.train(True)
       	epoch_time = time.time()
        loss_list = []
        tmp_loss_list = []
        for i , data in enumerate(trainLoader, 0):
            begin_time = time.time()
            optimizer.zero_grad()
            loss_miniBatch, output_albedo, output_normal, output_shading, \
                   diff_normal, coarse_normal = networkForward(data, log_weight, optimizer, Testing=False)
            numLoss = len(loss_miniBatch['loss_list'])
            loss = loss_miniBatch['albedo'].data[0] + loss_miniBatch['albedo_grad'].data[0] + \
                    loss_miniBatch['shading'].data[0] + loss_miniBatch['shading_grad'].data[0] + \
                    loss_miniBatch['normal'].data[0] + loss_miniBatch['normal_grad'].data[0] + \
                    loss_miniBatch['light'].data[0] + loss_miniBatch['loss_image'].data[0]  + \
                    log_weight['albedo']*loss_miniBatch['log_albedo'].data[0] + \
                    log_weight['shading']*loss_miniBatch['log_shading'].data[0] + \
                    log_weight['normal']*loss_miniBatch['log_normal'].data[0] + \
                    log_weight['image']*loss_miniBatch['log_image'].data[0]


            loss_list.append(loss_miniBatch['loss_list'])
            tmp_loss_list.append(loss_miniBatch['loss_list'])

            running_loss += loss
            running_albedo_loss  += loss_miniBatch['albedo'].data[0]
            running_shading_loss  += loss_miniBatch['shading'].data[0]
            running_normal_loss  += loss_miniBatch['normal'].data[0]
            running_light_loss  += loss_miniBatch['light'].data[0]
            running_image_loss  += loss_miniBatch['loss_image'].data[0]
            running_log_albedo += loss_miniBatch['log_albedo'].data[0]
            running_log_shading += loss_miniBatch['log_shading'].data[0]
            running_log_image += loss_miniBatch['log_image'].data[0]
            running_log_normal += loss_miniBatch['log_normal'].data[0]

            # record loss
            if i%monitor_count == monitor_count - 1:
                numIte_training = numIte_training + 1
                tmp_loss_list = np.array(tmp_loss_list)
                tmp_loss_list = np.mean(tmp_loss_list, axis=0)
                print '[%d %5d] loss: ' % (epoch + 1, i+1),
                print '%.4f '*numLoss % tuple(tmp_loss_list)
                print>>fid, '%d %5d ' % (epoch+1, i+1),
                print>>fid, '%.4f '*numLoss % tuple(tmp_loss_list)

                writer.add_scalar('train/loss_albedo', running_albedo_loss/monitor_count, numIte_training)
                writer.add_scalar('train/loss_shading', running_shading_loss/monitor_count, numIte_training)
                writer.add_scalar('train/loss_normal', running_normal_loss/monitor_count, numIte_training)
                writer.add_scalar('train/loss_lighting', running_light_loss/monitor_count, numIte_training)
                writer.add_scalar('train/loss_image', running_image_loss/monitor_count, numIte_training)
                writer.add_scalar('train/loss_log_albedo', running_log_albedo/monitor_count, numIte_training)
                writer.add_scalar('train/loss_log_shading', running_log_shading/monitor_count, numIte_training)
                writer.add_scalar('train/loss_log_image', running_log_image/monitor_count, numIte_training)
                writer.add_scalar('train/loss_log_normal', running_log_normal/monitor_count, numIte_training)
                writer.add_image('train/albedo', output_albedo)
                writer.add_image('train/shading', output_shading)
                writer.add_image('train/normal', output_normal)
                writer.add_image('train/normal_diff', diff_normal)
                writer.add_image('train/normal_coarse', coarse_normal)

                running_loss = 0
                running_albedo_loss = 0
                running_shading_loss = 0
                running_normal_loss = 0
                running_light_loss = 0
                running_image_loss = 0
                running_log_albedo = 0
                running_log_shading = 0
                running_log_image = 0
                running_log_normal = 0
                tmp_loss_list = []

            #print 'One iteration is %s second' % (time.time() - begin_time)
        loss_list = np.mean(np.array(loss_list), axis=0)
        print >> fid_sep, '%0.6f '* numLoss % tuple(loss_list)
        print '%0.6f '* numLoss % tuple(loss_list)
        # ---------------------------------------------------------
        # validation 
        # ---------------------------------------------------------
        my_network.train(False)
        test_loss = 0
        # loss for each component
        test_albedo_loss = 0
        test_shading_loss = 0
        test_normal_loss = 0
        test_light_loss = 0
        test_image_loss = 0
        test_log_albedo = 0
        test_log_shading = 0
        test_log_image = 0
        test_log_normal = 0

        count = 0
        test_loss_list = []
        for i, data in enumerate(testLoader, 0):
            loss_miniBatch, output_albedo, output_normal, output_shading, \
                    diff_normal, coarse_normal= networkForward(data, log_weight, optimizer, Testing=True)
            numLoss = len(loss_miniBatch['loss_list'])
            loss = loss_miniBatch['albedo'].data[0] + loss_miniBatch['albedo_grad'].data[0] + \
                    loss_miniBatch['shading'].data[0] + loss_miniBatch['shading_grad'].data[0] + \
                    loss_miniBatch['normal'].data[0] + loss_miniBatch['normal_grad'].data[0] + \
                    loss_miniBatch['light'].data[0] + loss_miniBatch['loss_image'].data[0] + \
                    log_weight['albedo']*loss_miniBatch['log_albedo'].data[0]+ \
                    log_weight['shading']*loss_miniBatch['log_shading'].data[0] + \
                    log_weight['normal']*loss_miniBatch['log_normal'].data[0] + \
                    log_weight['image']*loss_miniBatch['log_image'].data[0]

            test_loss_list.append(loss_miniBatch['loss_list'])
            test_loss += loss
            test_albedo_loss  += loss_miniBatch['albedo'].data[0]
            test_shading_loss  += loss_miniBatch['shading'].data[0]
            test_normal_loss  += loss_miniBatch['normal'].data[0]
            test_light_loss += loss_miniBatch['light']
            test_image_loss  += loss_miniBatch['loss_image'].data[0]
            test_log_albedo += loss_miniBatch['log_albedo']
            test_log_shading += loss_miniBatch['log_shading']
            test_log_image += loss_miniBatch['log_image']
            test_log_normal += loss_miniBatch['log_normal']
        
            count = count + 1
        test_loss_list = np.mean(np.array(test_loss_list), axis=0)
        print '[%d  ] test loss: ' % (epoch+1),
        print '%.4f '*numLoss % tuple(test_loss_list)
        print>>fid_test, '%d ' % (epoch+1),
        print>>fid_test, '%.4f '*numLoss % tuple(test_loss_list)
        
        writer.add_scalar('test/loss_albedo', test_albedo_loss/count, epoch)
        writer.add_scalar('test/loss_shading', test_shading_loss/count, epoch)
        writer.add_scalar('test/loss_normal', test_normal_loss/count, epoch)
        writer.add_scalar('test/loss_light', test_light_loss/count, epoch)
        writer.add_scalar('test/loss_image', test_image_loss/count, epoch)
        writer.add_scalar('test/loss_log_albedo', test_log_albedo/count, epoch)
        writer.add_scalar('test/loss_log_shading', test_log_shading/count, epoch)
        writer.add_scalar('test/loss_log_image', test_log_image/count, epoch)
        writer.add_scalar('test/loss_log_normal', test_log_normal/count, epoch)
        writer.add_image('test/albedo', output_albedo)
        writer.add_image('test/shading', output_shading)
        writer.add_image('test/normal', output_normal)
        writer.add_image('test/normal_diff', diff_normal)
        writer.add_image('test/normal_coarse', coarse_normal)
    
        print >> fid_test_sep, '%0.6f '* numLoss % tuple(test_loss_list)
        print '%0.6f '* numLoss % tuple(test_loss_list)

        tmp_saveName = os.path.join(saveIntermedia, 'trained_model_{:02d}.t7'.format(epoch))
        my_network.cpu()
        torch.save(my_network.state_dict(), tmp_saveName)
        my_network.cuda()
        print 'this epoch cost %s seconds' %(time.time() - epoch_time)

    my_network.cpu()
    torch.save(my_network, os.path.join(savePath, 'trained_model.t7'))
    print 'time used for training is %s' % (time.time() - begin_time)
    print('Finished training')
    
    fid.close()
    fid_test.close()
    fid_sep.close()
    fid_test_sep.close()

if __name__ == '__main__':
    savePath = sys.argv[1]
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    if len(sys.argv) > 2:
        log_weight = {}
        lr = float(sys.argv[2])
        weight_decay = float(sys.argv[3])
        total_epoch=int(sys.argv[4])
        load_path_coarse = sys.argv[5]
        load_path_fine = sys.argv[6]
        log_weight['albedo'] = float(sys.argv[7])
        log_weight['shading'] = float(sys.argv[8])
        log_weight['normal'] = float(sys.argv[9])
        log_weight['image'] = float(sys.argv[10])
        main(savePath, load_path_coarse, load_path_fine, log_weight, lr, weight_decay, total_epoch)
    else:
        main(savePath)
