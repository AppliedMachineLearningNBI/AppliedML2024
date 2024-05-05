#############################################################################
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import StandardScaler
import numpy
import cv2
import torch

#############################################################################

cols_mva = ['Area (ABD)', 'Area (Filled)', 'Aspect Ratio', 'Biovolume (Cylinder)',
       'Biovolume (P. Spheroid)', 'Circle Fit',
       'Circularity', 'Circularity (Hu)', 'Compactness', 'Convex Perimeter',
       'Convexity', 'Diameter (ABD)', 'Diameter (ESD)', 'Edge Gradient',
       'Elongation', 'Feret Angle Max', 'Feret Angle Min', 'Fiber Curl',
       'Fiber Straightness', 'Geodesic Aspect Ratio', 'Geodesic Length',
       'Geodesic Thickness', 'Intensity', 'Length', 'Particles Per Chain',
       'Perimeter', 'Roughness', 'Sigma Intensity', 'Sum Intensity',
       'Symmetry', 'Transparency', 'Volume (ABD)', 'Volume (ESD)', 'Width']


class ParticleDataset():
    
    def __init__(self, df, transform=None):
        
        self.df = df
        self.imgpaths = df['imgpaths'].to_numpy()
        self.labels = df['class'].to_numpy()
        self.transform = transform 
        scaler = StandardScaler()
        self.X_features = scaler.fit_transform(df[cols_mva])
        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        imgpath = self.imgpaths[idx]
        
        image = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
                    
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        label = torch.tensor(self.labels[idx]).int()
        xfeatures = torch.from_numpy(self.X_features[idx]).float()
        
        return image, label, imgpath, xfeatures


transform = A.Compose([
    A.Flip(p=0.5), 
    A.Resize(128, 128), 
    A.Normalize(mean=[94.,94.,94], std=[12.,12.,12], max_pixel_value=1.0),
    ToTensorV2()])











import numpy as np
import itertools
import matplotlib.pyplot as plt

# Confusion matrix function
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          savefig = ''):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=16)
        plt.yticks(tick_marks, target_names, fontsize=16)
        

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     fontsize=13,
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     fontsize=16,
                     color="white" if cm[i, j] > thresh else "black")


    #plt.tight_layout()
    plt.ylabel('True label', fontsize=16)
    plt.xlabel(f'Predicted label\n\n accuracy={accuracy:0.3f}; misclass={misclass:0.3f}', fontsize=16)
    cbar.ax.set_ylabel('Number of items',  labelpad=20, rotation=270, fontsize=16)   
    
    
    if savefig: plt.savefig(savefig, bbox_inches='tight')
    
    plt.show()