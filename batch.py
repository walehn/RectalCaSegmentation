import os, glob

data_path = "./data/"
data_dir = os.path.join(data_path, "nifti_data")
train_images = sorted(glob.glob(os.path.join(data_dir, "image", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "mask", "*.nii.gz")))

f = open("batch.csv","w")

for i in range(len(train_images)+1):
    if i==0:
        f.write('Image' + ',' + 'Mask' + '\n')
    else:
        f.write(train_images[i-1] + ',' + train_labels[i-1] + '\n')
f.close()