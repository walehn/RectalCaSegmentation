import os, glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology as mp

os.environ["MONAI_DATA_DIRECTORY"] = "./data"
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = directory
print(root_dir)

data_path = root_dir
data_dir = os.path.join(data_path, "nifti_data")
images = sorted(glob.glob(os.path.join(data_dir, "image", "*.nii.gz")))
labels = sorted(glob.glob(os.path.join(data_dir, "mask", "*.nii.gz")))


for i in range(len(images)) :
    t2 = nib.load(images[i])
    msk = nib.load(labels[i])

    t2img = t2.get_fdata()
    print('size of t2 image = ',t2img.shape)
    mskimg = msk.get_fdata()
    print('size of msk image = ',mskimg.shape)

    zzz=np.sum(np.sum(mskimg,axis=1),axis=0)
    zmax=np.argmax(zzz)
    print('slice # of largest mass:',zmax)

    n_slice = zmax # slice number to be displayed
    plt.figure("check", (20, 8))
    plt.imshow(np.concatenate((t2img[:,:,n_slice],1000*mskimg[:,:,n_slice]),axis=1))

    Ne = 2 # erosion의 정도를 결정
    Nd = 8 # dilation의 정도를 결정

    mskimg2 = mskimg>0.5
    mskimg2e = np.zeros(mskimg.shape)
    mskimg2d = np.zeros(mskimg.shape)
    NX,NY,NZ= mskimg.shape
    NZ
    # for 2d erosion/dilation (for conventional 2D MR images)
    for nz in range(NZ):
        mskimg2e[:,:,nz] = mp.binary_erosion(mskimg2[:,:,nz],selem=mp.disk(Ne))
        mskimg2d[:,:,nz] = mp.binary_dilation(mskimg2[:,:,nz],selem=mp.disk(Nd))

    plt.figure("check", (20, 8))
    plt.imshow(np.concatenate((mskimg2[:,:,n_slice],
    (mskimg2d[:,:,n_slice].astype(int)-mskimg2e[:,:,n_slice].astype(int))>0.5), axis=1))

    ## save
    mskimg_new = ((mskimg2d.astype(int)-mskimg2e.astype(int))>0.5).astype(int)
    msk_new=nib.Nifti1Image(mskimg_new,msk.affine,msk.header)

    text = labels[i]
    string = text.split('/')
    string2 = string[4].split('.nii')
    filename = string[0]+'/'+string[1]+'/'+string[2]+'/TB/'+string2[0]+"_peritumoral.nii.gz"
    print(filename)

    nib.save(msk_new,filename)
    plt.imshow(np.concatenate((t2img[:,:,n_slice],3000*mskimg[:,:,n_slice],3000*mskimg_new[:,:,n_slice]),axis=1))