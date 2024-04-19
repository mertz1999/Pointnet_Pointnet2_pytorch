"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
from torchvision import transforms
import utils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--model', type=str, required=True, help='model path')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()


def test(model, loader, num_class=40, vote_num=1):
    test_time_augmentation = transforms.Compose([
                                utils.RandRotation_z(),
                                utils.RandomNoise(),
                                utils.ToTensor()
                             ])
    # list of class names
    classes = [line.rstrip() for line in open('./modelnet40_normal_resampled/modelnet40_shape_names.txt')]

    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))

    total_labels = []
    total_predic = []


    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu: device = 'cuda:0'
        else: device = 'cpu'
    
        pred, _ = classifier(points.transpose(2, 1).to(device))

        # test time aug
        for _ in range(vote_num):
          aug_data = points.clone()
          for i in range(pred.shape[0]):
            aug_data[i] = test_time_augmentation(aug_data[i])

          aug_out, _ = classifier(aug_data.transpose(2, 1).to(device))
          pred += aug_out

        pred /= vote_num+1
        pred_choice = pred.data.max(1)[1]

        target = target.cuda()
        total_labels += list(target.detach().cpu().numpy())
        total_predic += list(pred_choice.detach().cpu().numpy())

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    instance_acc = np.mean(mean_correct)

    # plot 
    cm = confusion_matrix(total_labels, total_predic)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)

    # Adding labels and title
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)  # Optional: Adjust rotation of y labels if needed
    plt.savefig('cm.png')
    print('confusion matrix is saved in cm.png')


    return instance_acc


def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''DATA LOADING'''
    print('Load dataset ...')
    data_path = './modelnet40_normal_resampled/'

    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = 40
    model = importlib.import_module('pointnet_cls')

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(args.model)
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        instance_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
        print('Test Instance Accuracy: %f' % (instance_acc))


if __name__ == '__main__':
    args = parse_args()
    main(args)
