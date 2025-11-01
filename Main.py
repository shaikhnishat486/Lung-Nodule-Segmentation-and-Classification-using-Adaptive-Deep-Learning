import os
import SimpleITK as sitk
import pandas as pd
from numpy import matlib
from AOA import AOA
from DBO import DBO
from GEA import GEA
from Global_Vars import Global_Vars
from Image_Results import *
from Model_DenseASPP import Model_DenseASPP
from Model_MVit import Model_ViT
from Model_MobileNet import Model_MobileNet
from Model_RAN import Model_RAN
from Model_VGG16 import Model_VGG16
from Objfun import objfun, objfun_cls
from Plot_Results import *
from Proposed import Proposed
from SSA import SSA

No_of_Dataset = 2


def ReadImage(Filename):
    image = cv.imread(Filename)
    image = np.uint8(image)
    if len(image.shape) > 2:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.resize(image, (256, 256))
    return image


def readMhd(filename):
    # read mhd/raw image
    itkimage = sitk.ReadImage(filename)
    scan = sitk.GetArrayFromImage(itkimage)  # 3D image
    spacing = itkimage.GetSpacing()  # voxelsize
    origin = itkimage.GetOrigin()  # world coordinates of origin
    transfmat = itkimage.GetDirection()  # 3D rotation matrix
    return scan, spacing, origin, transfmat


# Read Dataset 1
an = 0
if an == 1:
    path = './Dataset/Dataset_1/LIDC-IDRI-slices'
    out_dir = os.listdir(path)
    Images = []
    Mask = []
    for i in range(len(out_dir)):
        fold = path + '/' + out_dir[i]
        in_dir = os.listdir(fold)
        for j in range(len(in_dir)):
            sub_fold = fold + '/' + in_dir[j]
            dir = os.listdir(sub_fold)
            files = sub_fold + '/' + 'images'
            files_dir = os.listdir(files)
            mask_files = sub_fold + '/' + 'mask-0'
            mask_dir = os.listdir(mask_files)
            for k in range(len(files_dir)):
                filename = files + '/' + files_dir[k]
                gtfile = mask_files + '/' + mask_dir[k]
                Image = ReadImage(filename)
                GT = ReadImage(gtfile)
                Images.append(Image)
                Mask.append(GT)
    np.save('Image_1.npy', np.asarray(Images))
    np.save('Ground_Truth_1.npy', np.asarray(Mask))

# Generate Target from GT 1
an = 0
if an == 1:
    Image = np.load('Image_1.npy', allow_pickle=True)
    GT = np.load('Ground_Truth_1.npy', allow_pickle=True)
    Target = []
    for i in range(len(GT)):
        GT1 = GT[i]
        if max(GT1.flatten()) == 0:
            Tar = 0
        else:
            Tar = 1
        Target.append(Tar)
    Target = np.reshape(Target, (-1, 1))
    np.save('Target_1.npy', Target)

# Read Dataset 2
an = 0
if an == 1:
    Image = []
    Mask = []
    Target = []
    df = pd.read_csv('Dataset/Dataset_2/trainset_csv/trainNodules.csv')
    IndID = df.values[:, 0].astype(int)
    RadID = df.values[:, 1].astype(int)
    FindingID = df.values[:, 2].astype(int)
    for s in range(len(IndID)):
        lnd = IndID[s]
        rad = RadID[s]
        finding = FindingID[s]
        [scan, spacing, origin, transfmat] = readMhd('Dataset/Dataset_2data0/LNDb-{:04}.mhd'.format(lnd))
        # Read segmentation mask
        [mask, spacing1, origin1, transfmat1] = readMhd('Dataset/Dataset_2masks/masks/LNDb-{:04}_rad{}.mhd'.format(lnd, rad))
        MASK = []
        for n in range(len(scan)):
            Org_Image = scan[n]
            mask_Image = mask[n]
            if len(np.unique(mask_Image)) > 1:
                Target.append(1)
                Image.append(Org_Image[n])
                Mask.append(mask_Image[n])
                MASK.append(mask_Image[n])
        i = 0
        while True:
            if len(np.unique(mask[i])) == 1:
                Target.append(0)
                Image.append(scan[:len(MASK)])
                Mask.append(mask[:len(MASK)])
            if i == len(MASK) - 1:
                break
            else:
                i += 1

    index = np.arange(len(Image))
    np.random.shuffle(index)
    Org_Img = np.asarray(Image)
    Shuffled_Datas = Org_Img[index]
    Shuffled_Target = Target[index]
    Shuffled_Mask = Mask[index]
    np.save('Target_2.npy', Shuffled_Target)
    np.save('Image_2.npy', Org_Img)
    np.save('Ground_Truth_2.npy', Shuffled_Mask)

# Segmentation
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Image = np.load('Image_' + str(n + 1) + '.npy', allow_pickle=True)
        Mask = np.load('Ground_Truth_' + str(n + 1) + '.npy', allow_pickle=True)
        Seg_Img = []
        Eval = []
        for i in range(len(Image)):
            Img = Image[i]
            Gt = Mask[i]
            eval, Seg = Model_DenseASPP(Img, Gt)
            Seg_Img.append(Seg)
            Eval.append(eval)
        np.save('Proposed_' + str(n + 1) + '.npy', Seg_Img)
        np.save('Seg_Eval.npy', Eval)

# Generate severity Classes
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Image = np.load('Proposed_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        GT = np.load('Ground_Truth_' + str(n + 1) + '.npy', allow_pickle=True)
        index = np.where(Target == 1)
        Org_Img = np.asarray(Image)
        Pred_Data = Org_Img[index[0]]
        Pred_Gt = GT[index[0]]
        np.save('Multi_Cls_Image_' + str(n + 1) + '.npy', Pred_Data)
        np.save('Multi_Cls_GT_' + str(n + 1) + '.npy', Pred_Gt)

# Generate Target Multiple classes
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Tar = []
        Ground_Truth = np.load('Multi_Cls_GT_' + str(n + 1) + '.npy', allow_pickle=True)
        for i in range(len(Ground_Truth)):
            image = Ground_Truth[i]
            if np.count_nonzero(image == 255) <= 100:
                Tar.append(1)  # Early Stage
            elif (np.count_nonzero(image == 255) > 100) & (np.count_nonzero(image == 255) <= 1000):
                Tar.append(2)  # Early Severe Stage
            elif (np.count_nonzero(image == 255) > 1000) & (np.count_nonzero(image == 255) <= 1500):
                Tar.append(3)  # Mid Stage
            else:
                Tar.append(4)  # Severe Stage
            # unique code
        df = pd.DataFrame(Tar)
        new_df = df.fillna(0)
        uniq = df[0].unique()
        Target = np.asarray(df[0])
        target = np.zeros((Target.shape[0], len(uniq)))  # create within rage zero values
        for uni in range(len(uniq)):
            index = np.where(Target == uniq[uni])
            target[index[0], uni] = 1
        np.save("Multi_Target_" + str(n + 1) + ".npy", target)

# Optimization for Binary Classification
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Data = np.load('Proposed_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Global_Vars.Data = Data
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 3  # Hidden Neuron Count, Number of Epochs, Learning Rate
        xmin = matlib.repmat([5, 5, 0.01], Npop, 1)
        xmax = matlib.repmat([255, 50, 0.99], Npop, 1)
        fname = objfun
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("SSA...")
        [bestfit1, fitness1, bestsol1, time] = SSA(initsol, fname, xmin, xmax, Max_iter)  # SSA

        print("DBO...")
        [bestfit2, fitness2, bestsol2, time1] = DBO(initsol, fname, xmin, xmax, Max_iter)  # DBO

        print("AOA...")
        [bestfit3, fitness3, bestsol3, time2] = AOA(initsol, fname, xmin, xmax, Max_iter)  # AOA

        print("GEA...")
        [bestfit4, fitness4, bestsol4, time3] = GEA(initsol, fname, xmin, xmax, Max_iter)  # GEA

        print("Proposed...")
        [bestfit5, fitness5, bestsol5, time4] = Proposed(initsol, fname, xmin, xmax, Max_iter)  # Proposed

        BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        np.save('BestSol_' + str(n + 1) + '.npy', BestSol)

# Binary Classification
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Feature = np.load('PROPOSED_' + str(n + 1) + '.npy', allow_pickle=True)  # loading step
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)  # loading step
        BestSol = np.load('BestSol_' + str(n + 1) + '.npy', allow_pickle=True)
        K = 5
        Per = 1 / 5
        Perc = round(Feature.shape[0] * Per)
        Fold = []
        for i in range(K):
            Eval = np.zeros((10, 25))
            for j in range(5):
                Feat = Feature
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Test_Data = Feat[i * Perc: ((i + 1) * Perc), :]
                Test_Target = Target[i * Perc: ((i + 1) * Perc), :]
                test_index = np.arange(i * Perc, ((i + 1) * Perc))
                total_index = np.arange(Feat.shape[0])
                train_index = np.setdiff1d(total_index, test_index)
                Train_Data = Feat[train_index, :]
                Train_Target = Target[train_index, :]
                Eval[j, :] = Model_ViT(Feature, Target, sol)  # Model_MViT With optimization
            Eval[5, :], pred = Model_RAN(Train_Data, Train_Target, Test_Data, Test_Target)  # Model RAN
            Eval[6, :], pred1 = Model_MobileNet(Train_Data, Train_Target, Test_Data, Test_Target)  # Model MobileNet
            Eval[7, :], pred2 = Model_VGG16(Train_Data, Train_Target, Test_Data, Test_Target)  # Model VGG16
            Eval[8, :], pred3 = Model_ViT(Feature, Target)  # Model_MViT Without optimization
            Eval[9, :] = Eval[4, :]
            Fold.append(Eval)
        np.save('Eval_all_KFold.npy', Fold)  # Save Eval all

# Optimization for severity Classification
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Data = np.load('Multi_Cls_Image_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Multi_Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Global_Vars.Data = Data
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 3  # Hidden Neuron Count, Number of Epochs, Learning Rate
        xmin = matlib.repmat([5, 5, 0.01], Npop, 1)
        xmax = matlib.repmat([255, 50, 0.99], Npop, 1)
        fname = objfun_cls
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("SSA...")
        [bestfit1, fitness1, bestsol1, time1] = SSA(initsol, fname, xmin, xmax, Max_iter)  # SSA

        print("DBO...")
        [bestfit2, fitness2, bestsol2, time2] = DBO(initsol, fname, xmin, xmax, Max_iter)  # DBO

        print("AOA...")
        [bestfit3, fitness3, bestsol3, time3] = AOA(initsol, fname, xmin, xmax, Max_iter)  # AOA

        print("GEA...")
        [bestfit4, fitness4, bestsol4, time4] = GEA(initsol, fname, xmin, xmax, Max_iter)  # GEA

        print("Proposed...")
        [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)  # Proposed

        BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        np.save('BestSol_' + str(n + 1) + '.npy', BestSol)

# severity Classification
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Feature = np.load('Multi_Cls_Image_' + str(n + 1) + '.npy', allow_pickle=True)  # loading step
        Target = np.load('Multi_Target_' + str(n + 1) + '.npy', allow_pickle=True)
        BestSol = np.load('BestSol_' + str(n + 1) + '.npy', allow_pickle=True)
        K = 5
        Per = 1 / 5
        Perc = round(Feature.shape[0] * Per)
        Fold = []
        for i in range(K):
            Eval = np.zeros((10, 25))
            for j in range(5):
                Feat = Feature
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Test_Data = Feat[i * Perc: ((i + 1) * Perc), :]
                Test_Target = Target[i * Perc: ((i + 1) * Perc), :]
                test_index = np.arange(i * Perc, ((i + 1) * Perc))
                total_index = np.arange(Feat.shape[0])
                train_index = np.setdiff1d(total_index, test_index)
                Train_Data = Feat[train_index, :]
                Train_Target = Target[train_index, :]
                Eval[j, :] = Model_ViT(Feature, Target, sol)  # Model_MViT With optimization
            Eval[5, :], pred = Model_RAN(Train_Data, Train_Target, Test_Data, Test_Target)  # Model RAN
            Eval[6, :], pred1 = Model_MobileNet(Train_Data, Train_Target, Test_Data, Test_Target)  # Model MobileNet
            Eval[7, :], pred2 = Model_VGG16(Train_Data, Train_Target, Test_Data, Test_Target)  # Model VGG16
            Eval[8, :], pred3 = Model_ViT(Feature, Target)  # Model_MViT Without optimization
            Eval[9, :] = Eval[4, :]
            Fold.append(Eval)
        np.save('Eval_all_KFold.npy', Fold)  # Save Eval all

plotConvResults()
Plot_Results()
plot_seg_results()
Plot_ROC_Curve()
Table()
Exiting_Table()
Image_Results()
Sample_Images()
