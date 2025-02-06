import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt_sne
from sklearn import datasets
from sklearn.manifold import TSNE
import os
import pandas as pd
 
def plot_tsne(features, labels, epoch,fileNameDir = None,rgb_colors=None):

    print(features.shape,labels.shape)
    print(type(features),type(labels))
    print(np.any(np.isnan(features)),np.any(np.isinf(features)))
    features = np.nan_to_num(features)
    if not os.path.exists(fileNameDir):
        os.makedirs(fileNameDir)
   
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    import seaborn as sns
 

    class_num = len(np.unique(labels)) 
    
    try:
        tsne_features = tsne.fit_transform(features) 
    except:
        tsne_features = tsne.fit_transform(features)
    
    
    df = pd.DataFrame()
    df["y"] = labels
    df["comp1"] = tsne_features[:, 0]
    df["comp2"] = tsne_features[:, 1]
    ax = plt_sne.gca()
    
    hex_colors = ['#%02x%02x%02x' % rgb for rgb in rgb_colors]

    sns.scatterplot(x= df.comp1.tolist(), y= df.comp2.tolist(),hue=df.y.tolist(),
                    palette=hex_colors,
                    data=df,
                    legend=False,ax=ax,s=10)
    
    ax.set_frame_on(True)              
    ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
    plt_sne.savefig("./tsne_CMM_UP.png")

    
from scipy.io import loadmat
if __name__ == '__main__':
    mat_feature = loadmat('feature/sc_test_features_all_UP.mat')  
    mat_label=loadmat('feature/sc_labels_UP.mat')
  
    features = mat_feature['test_features_all']

    labels = mat_label['labels']
    labels=labels.reshape(-1)
    print(features.shape)
    print(labels.shape)
    #rgb_colors=[(140, 67,46),(0,0,255),(255,100,0),(0, 255 ,200),(164,75,155),(101 ,174, 255),(118 ,254 ,254),( 60, 91 ,112),(255,255,0),(185 ,153, 185),(255, 0 ,255),(100, 0 ,255),(0 ,200, 254),(0, 255, 0),(171, 175, 80),(101, 193, 60)]
    rgb_colors=[(216 ,191 ,216),(0 , 255, 0),( 0 ,255, 255),(45, 138 ,86),(255 ,0, 255),(255, 165 ,0),(159, 31 ,239),(255, 0 ,0),(255,255,0)]
    plot_tsne(features, labels, "Set2", fileNameDir="test",rgb_colors=rgb_colors)