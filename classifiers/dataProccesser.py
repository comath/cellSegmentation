import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import os
from skimage.io import imread
from skimage.transform import resize

import matplotlib.pyplot as plt


dsb_data_dir = os.path.join('..','input')
stage_label = 'stage1'


train_labels = pd.read_csv(os.path.join(dsb_data_dir,'{}_train_labels.csv'.format(stage_label)))
train_labels['EncodedPixels'] = train_labels['EncodedPixels'].map(lambda ep: [int(x) for x in ep.split(' ')])


all_images = glob(os.path.join(dsb_data_dir, 'stage1_*', '*', '*', '*'))
img_df = pd.DataFrame({'path': all_images})
img_id = lambda in_path: in_path.split('/')[-3]
img_type = lambda in_path: in_path.split('/')[-2]
img_group = lambda in_path: in_path.split('/')[-4].split('_')[1]
img_stage = lambda in_path: in_path.split('/')[-4].split('_')[0]
img_df['ImageId'] = img_df['path'].map(img_id)
img_df['ImageType'] = img_df['path'].map(img_type)
img_df['TrainingSplit'] = img_df['path'].map(img_group)
img_df['Stage'] = img_df['path'].map(img_stage)

train_df = img_df.query('TrainingSplit=="train"')
train_rows = []
group_cols = ['Stage', 'ImageId']
for n_group, n_rows in train_df.groupby(group_cols):
    c_row = {col_name: col_value for col_name, col_value in zip(group_cols, n_group)}
    c_row['masks'] = n_rows.query('ImageType == "masks"')['path'].values.tolist()
    c_row['images'] = n_rows.query('ImageType == "images"')['path'].values.tolist()
    train_rows += [c_row]
train_img_df = pd.DataFrame(train_rows)    
IMG_CHANNELS = 3
shapes = []
def shapes(in_img_list):
    return imread(in_img_list[0]).shape

def read(in_img_list):
	return resize(imread(in_img_list[0])/255.0,[500,500,3])

ones = np.ones([500,500],dtype=np.int32)
zeros = np.zeros([500,500],dtype=np.int32)

def read_and_stack(in_img_list):
    masksOut = np.zeros([500,500],dtype=np.int32)
    for i,c_img in enumerate(in_img_list):
        mask = resize(imread(c_img),[500,500])
        oned_mask = np.where(mask > 0.001, ones, zeros)
        masksOut = masksOut + (i+1)*oned_mask
    return masksOut

def encode_rle(in_img_list):
    masksOut = np.zeros([400,500,2],dtype=np.int32)
    for i,c_img in enumerate(in_img_list):
        mask = resize(imread(c_img),[500,500])
        mask = np.where(mask == 1)
        print mask
        for j in range(500):
            y_0 = 0
            y_1 = 0
            for k in range(499):
                if mask[j,k] == 0 and mask[j,k+1] > 0:
                    y_0 = k
                if mask[j,k] > 0 and mask[j,k+1] == 0:
                    y_1 = k
            masksOut[i,j,0] = y_0
            masksOut[i,j,1] = y_1
    return masksOut

def read_and_sum(in_img_list):
    return np.sum(np.stack([imread(c_img) for c_img in in_img_list], 0), 0)/255.0

n_img = 6
print("Building Image Shapes")
train_img_df['image_shapes_x'] = train_img_df['images'].map(shapes).map(lambda x: x[0])
train_img_df['image_shapes_y'] = train_img_df['images'].map(shapes).map(lambda x: x[1])

print("Building Images")
train_img_df['images'] = train_img_df['images'].map(read).map(lambda x: x[:,:,:IMG_CHANNELS])


print("Counting Masks")
train_img_df['masksCount'] = train_img_df['masks'].map(len)
print("Building Run Length Encoded Masks")
train_img_df['rle_masks'] = train_img_df['masks'].map(encode_rle)


fig, m_axs = plt.subplots(2, n_img, figsize = (12, 4))
for (_, c_row), (c_im, c_lab) in zip(train_img_df.sample(n_img).iterrows(), 
                                     m_axs.T):
    c_im.imshow(c_row['images'])
    c_im.axis('off')
    c_im.set_title('Microscope')
    
    c_lab.imshow(c_row['rle_masks'])
    c_lab.axis('off')
    c_lab.set_title('Labeled')
    c_lab.set_title(str(c_row['masksCount']) + str(c_row['masks'].shape))

train_img_df.to_pickle("rle_cell.df")

print train_img_df['image_shapes_x'].mean()
print train_img_df['image_shapes_y'].mean()


