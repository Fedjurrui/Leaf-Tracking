import os, json
import numpy as np

from tqdm.notebook import tqdm
from PIL import Image, ImageDraw
from cv2 import findContours, RETR_TREE,CHAIN_APPROX_SIMPLE
from sklearn.metrics import jaccard_score

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage

class leafDataset(Dataset):
    def __init__(self,image_list,json_list):
        self.images = [torch.tensor(np.array(Image.open(img).convert("L")),dtype=torch.uint8) for img in image_list]
        self.images = torch.stack([torch.reshape(img,shape=(1,img.shape[0],img.shape[1])) for img in self.images])
        self.images = self.images/255
        self.targets = []
        self.ids_dict = {}
        count = 0
        for files in image_list:
            self.ids_dict[count] = os.path.split(files)[-1]
            count += 1
        for idx,img in tqdm(enumerate(self.images)):
            target = {
                "boxes":[],
                "labels":[],
                "iscrowd":[],
                "area":[],
                "image_id":torch.tensor(idx,dtype=torch.int64),
                "masks": []
            }
            with open(json_list[idx]) as o:
                jfile = json.load(o)
            for element in jfile["annotations"]:
                mask = Image.new(size=(img.shape[1],img.shape[2]),mode="L")
                draw = ImageDraw.Draw(mask)
                if "complex_polygon" in element.keys():
                    temp = []
                    for i in element["complex_polygon"]["path"]:
                        temp += [(e["x"],e["y"]) for e in i]
                    draw.polygon(temp,fill=(255))
                else:
                    draw.polygon([(e["x"],e["y"]) for e in element["polygon"]["path"]],fill=(255))
                mask = torch.tensor(np.array(mask)/255,dtype=torch.uint8)
                target["masks"].append(mask)
                pos = torch.where(mask)
                x0 = torch.min(pos[1])
                x1 = torch.max(pos[1])
                y0 = torch.min(pos[0])
                y1 = torch.max(pos[0])
                bbox = torch.tensor([x0,y0,x1,y1],dtype=torch.float32)
                target["boxes"].append(bbox)
                target["area"].append((x1-x0)*(y1-y0))
                target["iscrowd"].append(torch.tensor(0,dtype=torch.uint8))
                target["labels"].append(torch.tensor(1,dtype=torch.int64))
            target["boxes"] = torch.stack(target["boxes"])
            target["area"] = torch.stack(target["area"])
            target["iscrowd"] = torch.stack(target["iscrowd"])
            target["labels"] = torch.stack(target["labels"])  
            target["masks"] = torch.stack(target["masks"])         
            self.targets.append(target)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        return self.images[index],self.targets[index]
    def plotExampleWithMasks(self,item,alpha=120):
        img = ToPILImage()((self.images[item]*255).to(torch.uint8)).convert("RGB")
        masks = self.targets[item]
        draw = ImageDraw.Draw(img,"RGBA")
        for i in range(masks["masks"].shape[0]):
            colors = np.random.randint(0,255,size=3)
            colors = (colors[0].astype(int),colors[1].astype(int),colors[2].astype(int),alpha)
            outLineColors = (colors[0].astype(int),colors[1].astype(int),colors[2].astype(int),alpha+70)
            mask = masks["masks"][i].numpy()*255
            mask = mask.astype(np.uint8)
            contours, _ = findContours(mask,RETR_TREE,CHAIN_APPROX_SIMPLE)
            contours = [tuple(e[0]) for e in contours[0].tolist()]
            draw.polygon(contours,fill=colors,outline=outLineColors)
            box = masks["boxes"][i]
            box = [box[0],box[1],box[2],box[3]]

            draw.rectangle(box,outline=outLineColors)
            draw.text((box[0],box[1]),text="{}-{}".format("leaf",i+1))
        return img


def findPairs_for_evaluation(groundTruth,predictions,confidence=0.5,threshold=0.15):
    gt = [e.numpy() for e in groundTruth["masks"]]
    pred = []
    pairs = []
    ids_to_remove = []
    notDetected = []
    missDetection = []

    for e in predictions["masks"]:
        e = e.numpy()
        e[e > confidence] = 1
        e[e <= confidence] = 0
        pred.append(e)

    for idx,query in enumerate(gt):
        found = False
        for id,j in enumerate(pred):
            iou = jaccard_score(query,j,average="micro")
            if iou > threshold:
                match = {
                    "groundTruth":query,
                    "prediction":j,
                    "IoU":iou,
                    "type":groundTruth["Category"]
                }
                best_id = id
                found = True
        if found == True:
            pairs.append(match)
            ids_to_remove.append(best_id)
        else:
            notDetected.append({
                    "groundTruth":query,
                    "type":groundTruth["Category"]
                })
    pred = np.delete(pred,ids_to_remove,0)
    for i in pred:
        missDetection.append({
                    "prediction":i,
                    "type":groundTruth["Category"]
                })





    return [pairs,notDetected,missDetection]
