import numpy as np
import torch
from sklearn.metrics import jaccard_score
def extractMasks(detection):
    r = []
    for idx,val in enumerate(detection["scores"]):
        if val < 0.875:
            continue
        r.append(detection["masks"][idx][0])
    r = torch.stack(r)
    r = r.detach().numpy()
    return r

def findPairs(r1,r2):
    emptyMask = np.zeros(shape=(266,266))
    pairs = []
    markedIds = []

    for idx,query in enumerate(r1):
        query[query > 0.5] = 1
        query = np.uint8(query)
        if query.sum() == 0:
            pairs.append({
                "iou":0,
                "mask2":emptyMask,
                "mask1":emptyMask
            })
            continue
        best_match={

                "iou":0,
                "mask2":query,
                "mask1":query,
        }
        for id,j in enumerate(r2):
            j[j > 0.5] = 1
            j = np.uint8(j)
            iou = jaccard_score(query,j,average="micro")
            iou_comp = iou > best_match["iou"]
            if iou_comp:
                best_match["iou"]=iou
                best_match["mask2"] = j
                stored = id
        if best_match["iou"] > 0.1:
            pairs.append(best_match)
            markedIds.append(stored)
        else:
            pairs.append({
                "iou":0,
                "mask2":query,
                "mask1":query
            })

    idsToRemove = np.unique(markedIds).tolist()
    r2 = np.delete(r2,idsToRemove,0)

    for e in r2:
        if e.sum() == 0:
            continue
        e[e >0.5] = 1

        pairs.append({
            "iou":0,
            "mask2":np.uint8(e),
            "mask1":emptyMask
        })
    return pairs
def emptyDuplicates(pairs):
    masks = [e["mask2"] for e in pairs]
    emptyMask = np.zeros(shape=(266,266),dtype=np.uint8)
    for idx,val in enumerate(masks):
        if val.sum() == 0:
            continue
        checkList = masks[:idx]+masks[idx+1:]
        for id,element in enumerate(checkList):
            iou = jaccard_score(val,element,average="micro")
            if iou == 1:
                if pairs[idx]["iou"] >= pairs[id]["iou"]:
                    pairs[id]["mask2"] = pairs[id]["mask1"]
                    pairs[id]["iou"] = 0
                else:
                    pairs[idx]["mask2"] = pairs[id]["mask1"]
                    pairs[idx]["iou"] = 0
    return pairs
def IoU(y_true,y_pred):
    #TODO Legacy implementation. Might be removed.
    y_true[y_true >= 0.5] = 1
    y_true[y_true < 0.5] = 0
    y_true = np.reshape(y_true,[-1])
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    y_pred = np.reshape(y_pred,[-1])

    intersection = np.sum(y_true*y_pred)+1
    union = np.sum(y_true)+np.sum(y_pred) - intersection+1
    return intersection/union