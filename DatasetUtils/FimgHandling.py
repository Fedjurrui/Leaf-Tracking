import numpy as np
import struct
import torch

def openFimg(path, normalized=True):
    with open(path) as f:
        numpy_data = np.fromfile(f, np.dtype("B"))
    final_list = []
    for i in range(0, len(numpy_data[8:]), 4):
        final_list.append(struct.unpack("<f", numpy_data[i:i + 4])[0])
    final_list = np.array(final_list, dtype=np.float64)
    image = np.reshape(final_list, newshape=(1024, 1360))
    if normalized:
        return normalizeImg(image)
    else:
        return image
def cutTrayInIndividualImages(fimg_path,
                              normalized = False,
                              square_definition=[110,910,280,1080]):
    fimg = openFimg(fimg_path,normalized=normalized)
    fimg = fimg[square_definition[0]:square_definition[1],
           square_definition[2]:square_definition[3]]
    tray = []
    for i in range(1, 4):
        thisCutY0 = fimg.shape[1] // 3 * (i - 1)
        thisCutY1 = fimg.shape[1] // 3 * (i)
        for n in range(1, 4):
            thisCutX0 = fimg.shape[0] // 3 * (n - 1)
            thisCutX1 = fimg.shape[0] // 3 * n
            newImg = fimg[thisCutX0:thisCutX1, thisCutY0:thisCutY1]
            tray.append(newImg)
    return tray

def normalizeImg(img,torch_format = True):
    if torch_format == True:
        return torch.tensor((img - np.min(img)) / (np.max(img) - np.min(img)))
    else:
        return (img - np.min(img)) / (np.max(img) - np.min(img))

def prepareForModel(torch_list):
    torch_stack = torch.stack(torch_list)
    torch_tensor = torch.reshape(torch_stack,shape=(torch_stack.shape[0],1,torch_stack.shape[1],torch_stack.shape[2]))
    torch_tensor =torch_tensor.type(torch.float32)

    return torch_tensor
