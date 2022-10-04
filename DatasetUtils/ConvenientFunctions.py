import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from tracking.timeSeries import createMaskTimeSeries
from .PlotUtils import plotLeafTimeSeries
from .FimgHandling import normalizeImg
import gc
def create_time_series(tray,results,index=0):
    time_series = [e[index] for e in results]
    images = []
    for e in tray:
        img = e[index].copy()
        img[img < 0]= 0
        images.append(img)
    normalized_images = [normalizeImg(e.copy()) for e in images]
    time_series = createMaskTimeSeries(time_series)
    series = plotLeafTimeSeries(time_series, normalized_images)
    return time_series,series,images

def get_average_and_store(path,name,time_series,series,images,
                          type="Fmp",
                          line_plot=True,
                          gif=True,
                          save_masks=True,
                          save_fcam_values=True):

    route=os.path.join(path,name)
    os.makedirs(route,exist_ok=True)
    total_average= []
    for i in range(len(images)):
        av = []
        for e in range(time_series[i].shape[0]):
            img = images[i].copy()
            mask = time_series[i][e].copy()
            mask[mask < 0.85] = 0
            mask[mask >= 0.85] = 1
            mask = np.array(mask,dtype=bool)
            img[np.invert(mask)] = 0
            img_sum = np.sum(img)
            mask_sum = np.sum(mask)
            if np.sum(img) == 0:
                average = np.nan
            else:
                average = img_sum/mask_sum
            av.append(average)
        total_average.append(av)
    total_average = pd.DataFrame(total_average)
    total_average.index = [e+1 for e in total_average.index]
    if line_plot == True:
        plt.figure(figsize=(12,8))
        sns.lineplot(data=total_average)
        plt.xticks(total_average.index.tolist())
        plt.legend()
        plt.savefig(os.path.join(route,"{}_{}_graph.png".format(name,type)),dpi=300)
    if gif == True:
        series[0].save(os.path.join(route,'{}_{}_masks.gif'.format(name,type)), save_all=True, append_images=series,duration=280)
    if save_masks == True:
        np.savez_compressed(os.path.join(route,"{}_{}_Mask_series".format(name,type)),a=np.array(time_series,dtype="object"))
    if save_fcam_values == True:
        np.savez_compressed(os.path.join(route,"{}_{}_Images_series".format(name,type)),a=np.array(images,dtype="object"))
    total_average.to_csv(os.path.join(route,"{}_{}_Average_Flcam_values_per_leaf.tsv".format(name,type)),sep="\t")
    plt.clf()
    plt.close()
    gc.collect()

