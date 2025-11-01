import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

No_Of_dataset = 2


def Image_Results():
    I = [[304, 302, 309, 311, 301], [321, 329, 370, 408, 415]]
    for n in range(No_Of_dataset):
        Images = np.load('Image_' + str(n+1) + '.npy', allow_pickle=True)
        GT = np.load('Ground_Truth_' + str(n+1) + '.npy', allow_pickle=True)
        UNET = np.load('MobileUnet_' + str(n+1) + '.npy', allow_pickle=True)
        RESUNET = np.load('DenseUnet_' + str(n+1) + '.npy', allow_pickle=True)
        PROPOSED = np.load('Proposed_' + str(n+1) + '.npy', allow_pickle=True)
        for i in range(len(I[n])):
            plt.subplot(2, 3, 1)
            plt.title('Original')
            plt.imshow(Images[I[n][i]])
            plt.subplot(2, 3, 2)
            plt.title('GroundTruth')
            plt.imshow(GT[I[n][i]])
            plt.subplot(2, 3, 3)
            plt.title('MobileUnet')
            plt.imshow(UNET[I[n][i]])
            plt.subplot(2, 3, 4)
            plt.title('DenseUnet')
            plt.imshow(RESUNET[I[n][i]])
            plt.subplot(2, 3, 5)
            plt.title('PROPOSED')
            plt.imshow(PROPOSED[I[n][i]])
            plt.tight_layout()
            plt.show()


def Sample_Images():
    for n in range(No_Of_dataset):
        Orig = np.load('Image_' + str(n+1) + '.npy', allow_pickle=True)
        ind = [10, 50, 100, 115, 145, 155]
        fig, ax = plt.subplots(2, 3)
        plt.suptitle("Sample Images from Dataset " + str(n + 1))
        plt.subplot(2, 3, 1)
        plt.title('Image-1')
        plt.imshow(Orig[ind[0]])
        plt.subplot(2, 3, 2)
        plt.title('Image-2')
        plt.imshow(Orig[ind[1]])
        plt.subplot(2, 3, 3)
        plt.title('Image-3')
        plt.imshow(Orig[ind[2]])
        plt.subplot(2, 3, 4)
        plt.title('Image-4')
        plt.imshow(Orig[ind[3]])
        plt.subplot(2, 3, 5)
        plt.title('Image-5')
        plt.imshow(Orig[ind[4]])
        plt.subplot(2, 3, 6)
        plt.title('Image-6')
        plt.imshow(Orig[ind[5]])
        plt.show()


if __name__ == '__main__':
    Image_Results()
    Sample_Images()
