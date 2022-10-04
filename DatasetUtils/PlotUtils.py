from PIL import ImageDraw
from torchvision.transforms import ToPILImage
from cv2 import findContours, boundingRect, RETR_TREE, CHAIN_APPROX_SIMPLE
import numpy as np
import struct


def plotExampleWithMasks(img, masks, alpha=120):
    img = ToPILImage(mode="RGB")(img)
    draw = ImageDraw.Draw(img, "RGBA")
    for i in range(masks["masks"].shape[0]):
        colors = np.random.randint(0, 255, size=3)
        colors = (colors[0].astype(int), colors[1].astype(int), colors[2].astype(int), alpha)
        outLineColors = (colors[0].astype(int), colors[1].astype(int), colors[2].astype(int), alpha + 70)
        mask = masks["masks"][i].numpy() * 255

        mask = mask.astype(np.uint8)
        contours, _ = findContours(mask, RETR_TREE, CHAIN_APPROX_SIMPLE)
        contours = [tuple(e[0]) for e in contours[0].tolist()]
        if len(contours) < 2:
            continue
        draw.polygon(contours, fill=colors, outline=outLineColors)
        box = masks["boxes"][i]
        box = [box[0], box[1], box[2], box[3]]

        draw.rectangle(box, outline=outLineColors)
        draw.text((box[0], box[1]), text="leaf-{}".format(i + 1))
    return img


def plotResultsWithMasks(img, masks,alpha=120,text=True,rectangle=True):
    img = ToPILImage(mode="L")(img).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    for i in range(masks["masks"].shape[0]):
        if masks["scores"][i] < 0.95:
            continue

        colors = np.random.randint(0, 255, size=3)
        colors = (colors[0].astype(int), colors[1].astype(int), colors[2].astype(int), alpha)
        outLineColors = (colors[0].astype(int), colors[1].astype(int), colors[2].astype(int), alpha + 70)
        mask = masks["masks"][i].detach().cpu().numpy()
        mask = np.abs(mask) > 0.5
        mask = mask.astype(np.uint8)[0]
        contours, _ = findContours(mask, RETR_TREE, CHAIN_APPROX_SIMPLE)

        contours = [tuple(e[0]) for e in contours[0].tolist()]
        if len(contours) < 2:
            continue
        draw.polygon(contours, fill=colors, outline=outLineColors)
        box = masks["boxes"][i].detach().cpu()
        box = [box[0], box[1], box[2], box[3]]
        if rectangle == True:
            draw.rectangle(box, outline=outLineColors)
        if text == True:
            draw.text((box[0], box[1]), text="leaf-{}".format(i + 1))
    return img
def plotPairsWithMasks(img1,img2, pairs,alpha=120):
    img1 = ToPILImage(mode="L")(img1).convert("RGB")
    img2 = ToPILImage(mode="L")(img2).convert("RGB")
    draw1 = ImageDraw.Draw(img1, "RGBA")
    draw2 = ImageDraw.Draw(img2, "RGBA")
    for i in range(len(pairs)):

        colors = np.random.randint(0, 255, size=3)
        colors = (colors[0].astype(int), colors[1].astype(int), colors[2].astype(int), alpha)
        outLineColors = (colors[0].astype(int), colors[1].astype(int), colors[2].astype(int), alpha + 70)

        mask1 = pairs[i]["mask1"]
        mask2 = pairs[i]["mask2"]
        if type(mask1) != type(None):
            mask1 = np.abs(mask1) > 0.5
            mask1 = mask1.astype(np.uint8)
            contours1, _ = findContours(mask1, RETR_TREE, CHAIN_APPROX_SIMPLE)
            box1 = boundingRect(contours1[0])
            contours1 = [tuple(e[0]) for e in contours1[0].tolist()]
            if len(contours1) < 2:
                continue
            draw1.polygon(contours1, fill=colors, outline=outLineColors)
            box1 = [box1[0], box1[1], box1[0]+box1[2], box1[1]+box1[3]]
            draw1.rectangle(box1, outline=outLineColors)
            draw1.text((box1[0], box1[1]), text="leaf-{}".format(i + 1))
        mask2 = np.abs(mask2) > 0.5
        mask2 = mask2.astype(np.uint8)

        contours2, _ = findContours(mask2, RETR_TREE, CHAIN_APPROX_SIMPLE)

        box2 = boundingRect(contours2[0])

        contours2 = [tuple(e[0]) for e in contours2[0].tolist()]

        if len(contours2) < 2:
            continue

        draw2.polygon(contours2, fill=colors, outline=outLineColors)


        box2 = [box2[0], box2[1], box2[0]+box2[2], box2[1]+box2[3]]

        draw2.rectangle(box2, outline=outLineColors)
        draw2.text((box2[0], box2[1]), text="leaf-{}".format(i + 1))
    return img1,img2



def plotLeafTimeSeries(time_series,images,text=True,rectangle=True):
    final_images = []
    largest = time_series[-1].shape[0]
    colors = []
    outLineColors = []
    for i in range(largest):
        color = np.random.randint(0, 255, size=3)
        color = (color[0].astype(int), color[1].astype(int), color[2].astype(int), 120)
        colors.append(color)
        outLineColor = (color[0].astype(int), color[1].astype(int), color[2].astype(int), 190)
        outLineColors.append(outLineColor)

    for id,image in enumerate(images):
        image = ToPILImage(mode="L")(image).convert("RGB")
        draw = ImageDraw.Draw(image,mode="RGBA")
        for idx, mask in enumerate(time_series[id]):
            mask = np.abs(mask) > 0.5
            mask = mask.astype(np.uint8)
            if mask.sum() == 0:
                continue
            contours, _ = findContours(mask, RETR_TREE, CHAIN_APPROX_SIMPLE)
            box = boundingRect(contours[0])
            contours = [tuple(e[0]) for e in contours[0].tolist()]
            if len(contours) < 2:
                continue
            draw.polygon(contours, fill=colors[idx], outline=outLineColors[idx])
            box = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
            if rectangle == True:
                draw.rectangle(box, outline=outLineColors[idx])
            if text == True:
                draw.text((box[0], box[1]), text="leaf-{}".format(idx + 1))
        final_images.append(image)
    return(final_images)


