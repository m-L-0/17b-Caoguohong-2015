import numpy as np
import keras
import tensorflow as tf
from PIL import Image
from keras.models import load_model



class Dataset(object):
    def __init__(self, dtype='uint8', is_row_iamge=False):
        '''数据集
        
        Args:
            dtype: uint8 或 float32，uint8：每个像素值的范围是[0, 255];float32像素值范围是[0., 1.]
            is_row_image: 是否将3维图片展开成1维
        '''
        images = np.fromfile('./images/test_image.bin', dtype=np.uint8).reshape(-1, 28, 28, 1)
        print(images.shape)
        if dtype == 'uint8':
            self.images = images
        else:
            images = images.astype(np.float32) / 255.
            self.images = images
        if is_row_iamge:
            self.images = images.reshape([-1, 784])
        self.num_of_images = 6500
        self.offset = 0
        print('共6500张图片')

    def next_batch(self, batch_size=50):
        # 返回False表示以及没有样本
        # 注意：最后一个批次可能不足batch_size 所以推荐选择6500可以整除的batch_size
        if (self.offset + batch_size) <= self.num_of_images:
            self.offset += batch_size
            return self.images[self.offset - batch_size: self.offset]
        elif self.offset < self.num_of_images:
            return self.images[self.offset:]
        else:
            False

            # if __name__ == '__main__':
            #     images = Dataset()
            #     b_img = images.next_batch(2)
            #     print(b_img)
            # print(b_img.shape)


if __name__ == "__main__":
    # images = Dataset()
    # b_img = images.next_batch(2)

    # images = Dataset(is_row_iamge=True, dtype='fl')
    # b_img = images.next_batch(6500)
    # a = []
    # for i in b_img:
    #     i = np.reshape(i, (-1, 1, 28, 28))
    #     a.append(i)
    # f = open('../key.txt','w')
    # model = load_model('../resnet.h5')
    # for i in range(6500):
    #     pre = model.predict(a[i])
    #     pre = str(np.argmax(pre))
    #     f.writelines(pre+'\r')
    # f.close()

    images = Dataset(is_row_iamge=True, )
    b_img = images.next_batch(6500)
    a = []
    for i in b_img:
        i = np.reshape(i, (28, 28))
        a.append(i)

    for i in range(6500):
        img = Image.fromarray(a[i])
        img.save('./images/'+str(i+1)+'.bmp','bmp')

