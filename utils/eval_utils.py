import numpy as np
from utils.common_utils import torch2numpy
from sklearn.metrics import confusion_matrix, classification_report, top_k_accuracy_score
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
import plotly.express as px
from PIL import Image
import io


class ClassificationEvaluator:
    """Util class for evaluation networks."""
    def __init__(self, label_info):
        # Initialise empty data storage
        self.label_info = label_info
        self.num_labels = len(self.label_info.keys())
        self.task = 'multi' if self.num_labels > 2 else 'binary'         
        self.name_labels = {}
        for k,v in self.label_info.items():
            self.name_labels[v] = k 
        self.count_matrix = np.zeros((self.num_labels, self.num_labels))
        self.y_gt = []
        self.y_pred = []
        self.y_pred_raw = []
        self.update_count_matrix = False


    def feed(self, gt_labels, pred_labels):   
        """Trace model output and ground-truth"""
        gt_labels = torch2numpy(gt_labels)
        pred_labels = torch2numpy(pred_labels)
        pred_labels_temp = pred_labels.argmax(axis=1)

        self.y_gt.extend(gt_labels)
        self.y_pred.extend(pred_labels_temp)   
        self.y_pred_raw.extend(pred_labels)

                              
    def get_results(self, path_to_save=None):
        self.count_matrix = confusion_matrix(self.y_gt,
                                             self.y_pred)
        self.update_count_matrix = True        
        result = {}
        
        result['total_samples'] = np.sum(self.count_matrix)
        result['total_tp'] = np.sum(np.array([self.count_matrix[i,i] for i in range(0,self.num_labels)]))
        result['accuracy'] = result['total_tp']/result['total_samples']
        
        if self.task == 'multi':
            result['topk_2_accuracy'] = top_k_accuracy_score(self.y_gt,
                                                            self.y_pred_raw,
                                                            k=2)
            result['topk_3_accuracy'] = top_k_accuracy_score(self.y_gt,
                                                            self.y_pred_raw,
                                                            k=3)

        num_items_per_label = np.sum(self.count_matrix,axis=1,keepdims=True)
        num_items_per_label = np.where(num_items_per_label==0,1,num_items_per_label)
        distribution_matrix = self.count_matrix/num_items_per_label

        report = classification_report(self.y_gt, self.y_pred, digits = 3, zero_division=0)
        
        if path_to_save is not None:
            np.savez(path_to_save,
                     action_idx_to_name = self.name_labels,
                     distribution_matrix = distribution_matrix)
        return result, report, self.array_to_markdown(self.count_matrix)
    

    def draw(self, path=None):
        if self.update_count_matrix:
            df = pd.DataFrame(self.count_matrix,
                              index = [i for _, i in self.name_labels.items()],
                              columns = [i for _, i in self.name_labels.items()])

            ## Disable seanborn due to speed issue
            # plt.figure(figsize = (24, 24))
            # sn.heatmap(df, annot=True)
            # plt.savefig('output_confusion_matrix.png')
            # return sn.heatmap(df, annot=True).get_figure()

            fig = px.imshow(df)
            fig.update_layout(
                xaxis_title="Predicted label",
                yaxis_title="True label"
            )
            image_bytes = fig.to_image(format="png")
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image.save(path+'.png')
            image_array = np.array(image)
            image_array = np.transpose(image_array, (2, 0, 1)).astype(np.float32)
            return image_array
        else:
            assert False, "You haven't compute confusion matrix yet"

    
    def array_to_markdown(self, array):
        column_names = [i for _, i in self.name_labels.items()]
        df = pd.DataFrame(array, columns=column_names)
        sum = pd.DataFrame(df.sum(axis=0)).T
        df = pd.concat([df, sum], ignore_index=True)
        column_names.append('Total')
        df[''] = column_names
        df = df.set_index('')
        return df.to_markdown()


    # Unused but keep it for future use
    def matrix_to_str(self, matrix): 
        return np.array2string(matrix, formatter={'float_kind':lambda x: "%.2f" % x}).replace('[[', ' ').replace('[', '').replace(']', '')
    

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.vals = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.vals.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeters:
    def __init__(self):
        self.average_meters = {}

    def add_loss_value(self, loss_name, loss_val, n=1):
        if loss_name not in self.average_meters:
            self.average_meters[loss_name] = AverageMeter()
        self.average_meters[loss_name].update(loss_val, n=n)