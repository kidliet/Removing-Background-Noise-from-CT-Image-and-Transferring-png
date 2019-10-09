import os
import SimpleITK
import dicom
import numpy as np
import cv2
import glob
from tqdm import tqdm
import itk
ImageType2dC=itk.Image[itk.UC,2]
ImageType2dS=itk.Image[itk.SS,2]
ImageType2dF=itk.Image[itk.F,2]


def body_slice(image):

    image=itk.GetImageFromArray(image.astype(np.float32))

    ctBackgroundRange=5
    ctBackgroundValue=-200
    #morphlogicalCloseRadius=20

#  sigmoid function to increase the contrast near ctBackgroundValue
    sigmoidImageFilter=itk.SigmoidImageFilter[ImageType2dF,ImageType2dF].New()
    sigmoidImageFilter.SetInput(image)
    sigmoidImageFilter.SetAlpha(-ctBackgroundRange)
    sigmoidImageFilter.SetBeta(ctBackgroundValue)
    sigmoidImageFilter.SetOutputMaximum(1)
    sigmoidImageFilter.SetOutputMinimum(1e-3)


    # fast marching begin form the four corners
    fastMarchingFilter=itk.FastMarchingImageFilter[ImageType2dF,ImageType2dF].New()

    fastMarchingFilter.SetInput(sigmoidImageFilter.GetOutput())
    # define four nodes
    size=image.GetLargestPossibleRegion().GetSize()
    spacing=image.GetSpacing()
    points=[(0,0),(0,size[1]-1),(size[0]-1,0),(size[0]-1,size[1]-1)]
    NodeType=itk.LevelSetNode.F2
    NodeContainer=itk.VectorContainer[itk.UI,NodeType]
    container=NodeContainer.New()
    container.Initialize()
    for i,p in enumerate(points):
        node=NodeType()
        node.SetIndex(p)
        node.SetValue(0.0)
        container.InsertElement(i,node)
    fastMarchingFilter.SetTrialPoints(container)
        # get the binary mask
    thresholdFilter=itk.BinaryThresholdImageFilter[ImageType2dF,ImageType2dC].New()
    thresholdFilter.SetInput(fastMarchingFilter.GetOutput())
    thresholdFilter.SetLowerThreshold(0)
    thresholdFilter.SetUpperThreshold(10000)
    thresholdFilter.SetOutsideValue(1)
    thresholdFilter.SetInsideValue(0)
    maskBody=itk.GetArrayFromImage(thresholdFilter.GetOutput())
    return maskBody


def load_patient(src_dir):
    '''
        读取某文件夹内的所有dicom文件
    :param src_dir: dicom文件夹路径
    :return: dicom list
    '''
    files = os.listdir(src_dir)
    print ('files:',files)
    slices = []
    for s in files:
        if is_dicom_file(src_dir + '/' + s):
            instance = dicom.read_file(src_dir + '/' + s)
            slices.append(instance)
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
    return slices
def get_pixels_hu_by_simpleitk(dicom_dir):
    '''
        读取某文件夹内的所有dicom文件,并提取像素值(-4000 ~ 4000)
    :param src_dir: dicom文件夹路径
    :return: image array
    '''
    reader = SimpleITK.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    img_array = SimpleITK.GetArrayFromImage(image)
    #max ,min = np.max(img_array),np.min(img_array)
    #print ('img_array',np.max(img_array),np.min(img_array))
    #img_array[img_array == -2000] = 0
    return img_array
def normalize_hu(image):
    '''
           将输入图像的像素值(-1000 ~ 400)归一化到0~1之间
       :param image 输入的图像数组
       :return: 归一化处理后的图像数组
    '''

    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    #image[image < -1000] = 0.
    image = (image - MIN_BOUND)/(MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image
def ospath(root):
    dataset = glob.glob(os.path.join('%s' % root, '*'))
    dataset.sort()
    dataset = dataset[:10000]
    return dataset
def itk_image_series_reader(dicom_path):
    image_type = itk.Image[itk.F, 2]
    reader = itk.ImageFileReader[image_type].New()
    reader.SetFileName(dicom_path)
    reader.Update()
    itk_image = reader.GetOutput()
    return itk_image
if __name__ == '__main__':
    dicom_dir = 'E:/data-101/CT_dim/'
    dataset = ospath(dicom_dir)
    count = 0
    #print (len(dataset),dataset)
    for i in tqdm(range(len(dataset))):
        img_path = "E:/data-101/CT_png/" + str(count).rjust(5, '0') + ".png"
        ds = itk_image_series_reader(dataset[i])  # read dicom
        ds = itk.GetArrayFromImage(ds)  #get array
        org_img = normalize_hu(ds)    # normalization
        #print (org_img)
        maskbody = body_slice(ds)     # get mask (0,1)
        final_img = (org_img*255)*maskbody
        #final_img = final_img.astype(np.int16)
        cv2.imwrite(img_path, final_img)
        count +=1
