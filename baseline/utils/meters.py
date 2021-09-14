import os
import json
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)

def average_accuracy(ious, save_fig=False, output_dir='./output', filename='accuracy_results'):
    accuracy = []
    average_accuracy = 0
    thresholds = np.arange(0, 1, 0.00001)
    for t in thresholds:
        predictions = (ious >= t).astype(int)
        TP = np.sum(predictions)
        a = TP / len(predictions)
        
        accuracy.append(a)

    for i, t in enumerate(zip(thresholds[:-1], thresholds[1:])): 
        average_accuracy += (np.abs(t[1]-t[0])) * accuracy[i]

    if save_fig:
        if not osp.exists(output_dir):
            os.mkdir(output_dir)
        save_json(osp.join(output_dir, '{:s}.json'.format(filename)), {'accuracy': accuracy})
        
        plt.plot(thresholds, accuracy)
        plt.xlim(0, 1)
        plt.xlabel('IoU')
        plt.ylim(0, 1)
        plt.ylabel('Accuracy')
        plt.title('Accuracy-IoU curve. AA={:.5f}'.format(average_accuracy))
        plt.savefig(osp.join(output_dir, '{:s}_curve.png'.format(filename)))
    
    return average_accuracy