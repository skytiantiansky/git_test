import os
import sys
import random
import argparse
import numpy as np
from PIL import Image, ImageFile
import face_recognition

#口罩图片文件
BLUE_IMAGE_PATH = r"./images/blue-mask.png"

def create_mask(image_path):
    #人脸图片路径
    pic_path = image_path
    #口罩图片路径
    mask_path = BLUE_IMAGE_PATH
    # mask_path = "/media/preeth/Data/prajna_files/mask_creator/face_mask/images/blue-mask.png"
    show = False
    model = "hog"
    FaceMasker(pic_path, mask_path, show, model).mask()

class FaceMasker:
    #鼻梁nose_bridge，脸颊chin 的特征点
    KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')

    def __init__(self, face_path, mask_path, show=False, model='hog'):
        self.face_path = face_path
        self.mask_path = mask_path
        self.show = show
        self.model = model
    #     self._face_img: ImageFile = None
    #     self._mask_img: ImageFile = None

    def mask(self):
        """
        图像载入函数 load_image_file load_image_file(file, mode='RGB')
            加载一个图像文件到一个numpy array类型的对象上。
            参数：
                file：待加载的图像文件名字
                mode：转换图像的格式 只支持“RGB”(8位RGB, 3通道)和“L”(黑白)
            返回值： 一个包含图像数据的numpy array类型的对象
        """
        face_image_np = face_recognition.load_image_file(self.face_path)
        """
        face_locations(img, number_of_times_to_upsample=1, model='hog') 
        给定一个图像，返回图像中每个人脸的面部特征位置(眼睛、鼻子等)，也即是获取每个人脸所在的边界框/人脸的位置(top, right, bottom, left)。
        参数： 
            img：一个image（numpy array类型）
            number_of_times_to_upsample：从images的样本中查找多少次人脸，该参数值越高的话越能发现更小的人脸。 
            model：使用哪种人脸检测模型。“hog” 准确率不高，但是在CPUs上运行更快，
                   “cnn” 更准确更深度（且 GPU/CUDA加速，如果有GPU支持的话），默认是“hog” 
            返回值： 一个元组列表，列表中的每个元组包含人脸的位置(top, right, bottom, left)
        """
        face_locations = face_recognition.face_locations(face_image_np, model=self.model)
        """
        人脸特征提取函数 face_landmarks(face_image, face_locations=None,model="large") 
            给定一个图像，提取图像中每个人脸的脸部特征位置。
            人脸特征提取函数face_landmarks 提取后的脸部特征包括：
                鼻梁nose_bridge、鼻尖nose_tip、 下巴chin、左眼left_eye、右眼right_eye、左眉 left_eyebrow、
                右眉right_eyebrow、上唇top_lip、下 唇bottom_lip
            参数： 
                face_image：输入的人脸图片 
                face_locations=None： 
                    可选参数，默认值为None，代表默认解码图片中的每一个人脸。 
                    若输入face_locations()[i]可指定人脸进行解码 
                model="large"/"small"：
                    输出的特征模型，默认为“large”，可选“small”。 
                    当选择为"small"时，只提取左眼、右眼、鼻尖这三种脸部特征。
        """
        face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)
        #实现array到image的转换,返回image
        self._face_img = Image.fromarray(face_image_np)#人脸图片
        self._mask_img = Image.open(self.mask_path) #口罩图片

        found_face = False #是否找到脸的标志符
        #遍历检测到的每个人脸所属的整个脸部特征
        for face_landmark in face_landmarks:
            #   {'chin': [ (1108, 361)], 'left_eyebrow':[...], 'right_eyebrow': [...],
            #   'nose_bridge': [...], 'nose_tip': [...], 'left_eye': [...], 'right_eye': [...],
            #   'top_lip': [...], 'bottom_lip':[...] }
            # print(face_landmark)
            # print(type(face_landmark)) #<class 'dict'>

            #鼻梁nose_bridge、脸颊chin 是否在 所检测出来的人脸特征字典中的标志
            skip = False
            #遍历 鼻梁nose_bridge、脸颊chin 的特征点
            for facial_feature in self.KEY_FACIAL_FEATURES:
                #判断 鼻梁nose_bridge、脸颊chin 是否在 所检测出来的人脸特征字典中
                if facial_feature not in face_landmark:
                    # 鼻梁nose_bridge、脸颊chin 不在 所检测出来的人脸特征字典中，则认为跳过不处理当前该检测出来的人脸特征
                    skip = True
                    break
            if skip:
                # 鼻梁nose_bridge、脸颊chin 不在 所检测出来的人脸特征字典中，则认为跳过不处理当前该检测出来的人脸特征
                continue

            # 找到脸
            found_face = True
            #传入当前所检测出来的人脸的整个脸部特征字典数据，进行对人脸中的口部位置贴上口罩
            self._mask_face(face_landmark)

        if found_face:
            if self.show:
                self._face_img.show()
            # save
            self._save()
        else:
            print('Found no face.')

    """
    鼻梁nose_bridge中的第2个特征点到脸颊chin中的第9个特征点(即下巴)构成一条竖直线；
    计算左脸颊chin中的第1个特征点到竖直线的距离乘以width_ratio作为左脸中口罩的宽，竖直线两端点坐标之间的距离计算2范数的值作为左脸中口罩的高；
    计算右脸颊chin中的第17个特征点到竖直线的距离乘以width_ratio作为右脸中口罩的宽，竖直线两端点坐标之间的距离计算2范数的值作为右脸中口罩的高；
    """
    # 传入当前所检测出来的人脸的整个脸部特征字典数据，进行对人脸中的口部位置贴上口罩
    def _mask_face(self, face_landmark: dict):
        """
        获取鼻梁nose_bridge中的第2个特征点，即获取整个人脸68个特征点中的第29个特征点。
        获取脸颊chin中的第1个特征点，即获取整个人脸68个特征点中的第1个特征点。
        获取脸颊chin中的第9个特征点，即获取整个人脸68个特征点中的第9个特征点。
        获取脸颊chin中的第17个特征点，即获取整个人脸68个特征点中的第17个特征点。

        整个人脸68个特征点中 第29个特征点 和 第1个特征点 和 第17个特征点 三个点一起构成了一条线段(直线)
        """
        #获取字典中鼻梁nose_bridge的特征数据：'nose_bridge': [...]。鼻梁有4个特征点。
        nose_bridge = face_landmark['nose_bridge']
        # print("nose_bridge",nose_bridge) # 比如 [(118, 63), (118, 75), (117, 88), (117, 101)]
        # 因为鼻梁有4个特征点，因此'nose_bridge': [...] 中的列表有4个元祖的特征值
        # nose_bridge[4 * 1 // 4] 即等于 nose_bridge[1] 取第2个元祖的特征值，
        # 也即是取出鼻梁nose_bridge中的第2个特征点的值，该特征点位于整个人脸68个特征点中的第29个
        nose_point = nose_bridge[len(nose_bridge) * 1 // 4]
        # print("nose_point",nose_point) # 比如 (118, 75)
        nose_v = np.array(nose_point) #把元祖封装的类型 转换为 array类型
        # print("nose_v", nose_v) # 比如 [118 75]

        #获取字典中脸颊chin的特征数据：'chin': [...]。脸颊chin有17个特征点。
        chin = face_landmark['chin']
        # chin_len = len(chin)
        # print(chin_len) #17
        #获取 脸颊chin中的 第9个特征点的值，该特征点位于整个人脸68个特征点中的第9个，即下巴底部位置
        chin_bottom_point = chin[8]
        # print("chin_bottom_point",chin_bottom_point) # 比如 (155, 341)
        # chin_bottom_point = chin[chin_len // 2]
        chin_bottom_v = np.array(chin_bottom_point) #把元祖封装的类型 转换为 array类型
        # print("chin_bottom_v",chin_bottom_v) # 比如 [155 341]
        # chin_left_point = chin[chin_len // 8]
        # chin_right_point = chin[chin_len * 7 // 8]
        # 获取 脸颊chin中的 第1个特征点的值，该特征点位于整个人脸68个特征点中的第1个
        chin_left_point = chin[0]
        # 获取 脸颊chin中的 第17个特征点的值，该特征点位于整个人脸68个特征点中的第17个
        chin_right_point = chin[16]

        # 分割口罩图片并调整大小
        width = self._mask_img.width  #口罩图片的width
        height = self._mask_img.height #口罩图片的height
        width_ratio = 1.2 #宽度比

        """
        1.nose_v - chin_bottom_v
            nose_v：鼻梁nose_bridge中的第2个特征点的值，该特征点位于整个人脸68个特征点中的第29个。
            chin_bottom_v：脸颊chin中的 第9个特征点的值，该特征点位于整个人脸68个特征点中的第9个，即下巴底部位置。
            nose_v - chin_bottom_v：(鼻梁)第29个特征点的值  - (下巴)第9个特征点的值
        2.x_norm = np.linalg.norm(x, ord=None, axis=None, keepdims=False) 求范数
            x: 表示矩阵
            ord：范数类型    
                ord=None(默认2范数)：矩阵中所有元素值的平方和(先对每个元素自身平方运算后再所有结果值求和)，再整体开根号
                ord=1(1范数)：每一列中所有列值求总和，获取出最大的某一列列值总和，即哪一列的列值总和是最大的
                ord=2(2范数)：矩阵中所有元素值的平方和，再整体开根号
                ord=np.inf(无穷范数)：每一行中所有行值求总和，获取出最大的某一行行值总和，即哪一行的行值总和是最大的
            axis：处理类型    
                axis=1表示按行向量处理，求多个行向量的范数
                axis=0表示按列向量处理，求多个列向量的范数
                axis=None表示矩阵范数。
        3.比如 nose_v - chin_bottom_v 的值为 [6 -147] 
          np.linalg.norm(nose_v - chin_bottom_v)
                ord=None(默认2范数)：矩阵中所有元素值的平方和(先对每个元素自身平方运算后再所有结果值求和)，再整体开根号
                因此 sqrt(6^2 + (-147)^2) = 147.1223980228707
        4.最终把 147.1223980228707 作为 切割出来的左边一半的口罩图片的高
        """
        # print("nose_v - chin_bottom_v",nose_v - chin_bottom_v) #比如 [6 -147] 147.1223980228707
        # print("int(np.linalg.norm(nose_v - chin_bottom_v))",int(np.linalg.norm(nose_v - chin_bottom_v))) #比如 147
        # 最终把 147.1223980228707 作为 切割出来的左边一半的口罩图片的高
        new_height = int(np.linalg.norm(nose_v - chin_bottom_v))

        # 计算左半边脸的口罩
        # 切割出口罩图片中左边的一半：(234 // 2, 146) 得到 (117, 146)
        # (0, 0, width // 2, height)：(0, 0) 代表 左上角的x/y，(width // 2, height) 代表 右下角的x/y
        mask_left_img = self._mask_img.crop((0, 0, width // 2, height))
        # print("_mask_img.size",self._mask_img.size) #(234, 146)
        # print("mask_left_img.size",mask_left_img.size) #(117, 146)

        """
        第29个特征点(x1,y1) 到 第9个特征点(x2,y2) 构成一条直线，计算 第1个特征点(x,y) 到 这条直线 的距离 作为 切割出来的左边一半的口罩图片的宽。
            第1个特征点(x,y)：chin_left_point，即脸颊chin中的 第1个特征点的值，该特征点位于整个人脸68个特征点中的第1个
            第9个特征点(x2,y2)：chin_bottom_point，即脸颊chin中的 第9个特征点的值，该特征点位于整个人脸68个特征点中的第9个，即下巴底部位置
            第29个特征点(x1,y1)：nose_point，即鼻梁nose_bridge中的第2个特征点的值，该特征点位于整个人脸68个特征点中的第29个
        """
        # 将 第1个特征点(x,y) 到 这条直线 的距离 作为 切割出来的左边一半的口罩图片的宽。
        mask_left_width = self.get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)

        #mask_left_width(第1个特征点(x,y) 到 这条直线 的距离) * width_ratio(宽度比1.2)
        mask_left_width = int(mask_left_width * width_ratio)
        # print("mask_left_width",mask_left_width) #比如 156

        # 将切割出来的左边一半的口罩图片 从 (117, 146) resize为 举例的 (156, 147)
        # 也即是把口罩左半边宽和高 按比例缩放到 左脸面颊的宽 和 鼻梁到下巴的高，将口罩左半边按比例缩放进行适配 左脸面颊的宽 和 鼻梁到下巴的高，
        # 为的就是能在人脸左脸为侧脸的时候仍能把口罩左半边按比例缩放进行适配给为侧脸的左脸。
        mask_left_img = mask_left_img.resize((mask_left_width, new_height))

        #  切割出口罩图片中右边的一半：(234 // 2, 146) 得到 (117, 146)
        # (width // 2, 0, width, height)：(width // 2, 0) 代表 左上角的x/y，(width, height) 代表 右下角的x/y
        mask_right_img = self._mask_img.crop((width // 2, 0, width, height))
        # print("mask_right_img.size",mask_right_img.size) #(117, 146)

        """
        第29个特征点(x1,y1) 到 第9个特征点(x2,y2) 构成一条直线，计算 第17个特征点(x,y) 到 这条直线 的距离 作为 切割出来的右边一半的口罩图片的宽。
            第17个特征点(x,y)：chin_right_point，即脸颊chin中的 第17个特征点的值，该特征点位于整个人脸68个特征点中的第17个
            第9个特征点(x2,y2)：chin_bottom_point，即脸颊chin中的 第9个特征点的值，该特征点位于整个人脸68个特征点中的第9个，即下巴底部位置
            第29个特征点(x1,y1)：nose_point，即鼻梁nose_bridge中的第2个特征点的值，该特征点位于整个人脸68个特征点中的第29个
        """
        mask_right_width = self.get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)

        # mask_right_width(第17个特征点(x,y) 到 这条直线 的距离) * width_ratio(宽度比1.2)
        mask_right_width = int(mask_right_width * width_ratio)
        # print("mask_right_width",mask_right_width) #比如 132

        # 将切割出来的右边一半的口罩图片 从 (117, 146) resize为 举例的 (132, 147)
        # 也即是把口罩右半边宽和高 按比例缩放到 右脸面颊的宽 和 鼻梁到下巴的高，将口罩右半边按比例缩放进行适配 右脸面颊的宽 和 鼻梁到下巴的高，
        # 为的就是能在人脸右脸为侧脸的时候仍能把口罩右半边按比例缩放进行适配给为侧脸的右脸。
        mask_right_img = mask_right_img.resize((mask_right_width, new_height))

        # 合并 左脸的左半边口罩 和 右脸的右半边口罩：(左半边口罩的宽width + 右半边口罩的宽width, 口罩的高new_height)
        size = (mask_left_img.width + mask_right_img.width, new_height)
        print("size",size) #比如 (548, 307)
        """
        paste(透明图层的图, (粘贴的起始位置))：粘贴后的图像的背景图层会变成黑色，一般不使用
        paste(透明图层的图, (粘贴的起始位置), 透明图层的图)：粘贴后的图像的背景图层依旧为透明，一般使用该方式
        """
        #RGBA：A代表透明图层通道
        mask_img = Image.new('RGBA', size)
        #paste(透明图层的左半边口罩的图, (0, 0), 透明图层的左半边口罩的图)：粘贴后的左半边口罩的图的背景图层依旧为透明
        mask_img.paste(mask_left_img, (0, 0), mask_left_img)
        #paste(透明图层的右半边口罩的图, (左半边口罩的宽width, 0), 透明图层的右半边口罩的图)：粘贴后的右半边口罩的图的背景图层依旧为透明
        mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

        """
        1.chin_bottom_point：脸颊chin中的第9个特征点(x2,y2)的值，该特征点位于整个人脸68个特征点中的第9个，即下巴底部位置
          nose_point：鼻梁nose_bridge中的第2个特征点的值，该特征点位于整个人脸68个特征点中的第29个特征点(x1,y1)
        2.np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
          => np.arctan2(脸颊chin中的第9个特征点(下巴底部位置)的y - 鼻梁nose_bridge中的第2个特征点的y, 
                        脸颊chin中的第9个特征点(下巴底部位置)的x - 鼻梁nose_bridge中的第2个特征点的x)
        3.numpy.arctan2(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) = <ufunc 'arctan2'>
            x1 :类似array, 实值。填入y坐标。
            x2 :类似array, 实值。填入x坐标。x2必须可广播以匹配x1的形状，反之亦然。
            
            arctan2与基础C库的atan2函数相同。C标准中定义了以下特殊值：
                x1	x2	arctan2(x1,x2)
                +/- 0	+0	+/- 0
                +/- 0	-0	+/- pi
                > 0	+/-inf	+0 / +pi
                < 0	+/-inf	-0 / -pi
                +/-inf	+inf	+/- (pi/4)
                +/-inf	-inf	+/- (3*pi/4)
            请注意，+ 0和-0是不同的浮点数，+inf和-inf也是如此。
            
            例子，考虑不同象限中的四个点：
                >>> x = np.array([-1, +1, +1, -1])
                >>> y = np.array([-1, -1, +1, +1])
                >>> np.arctan2(y, x) * 180 / np.pi
                array([-135.,  -45.,   45.,  135.])
                注意参数的顺序。arctan2在x2 = 0时以及在其他几个特殊点处也定义，获得以下范围内的值[-pi, pi]
                >>> np.arctan2([1., -1.], [0., 0.])
                array([ 1.57079633, -1.57079633])
                >>> np.arctan2([0., 0., np.inf], [+0., -0., np.inf])
                array([ 0.        ,  3.14159265,  0.78539816])
        """
        # print("chin_bottom_point[1]",chin_bottom_point[1]) #比如 107
        # print("nose_point[1]",nose_point[1])#比如 66
        # print("chin_bottom_point[0]",chin_bottom_point[0])#比如 84
        # print("nose_point[0]",nose_point[0]) #比如 80
        # print("chin_bottom_point[1] - nose_point[1]",chin_bottom_point[1] - nose_point[1])#比如 41
        # print("chin_bottom_point[0] - nose_point[0]",chin_bottom_point[0] - nose_point[0])#比如 4

        """
        radian = np.arctan2(y值,x值) 计算两点之间的弧度
            参数：
                y值：两个点y值之间差值。
                x值：两个点x值之间差值。
        把两点之间计算出来的弧度转换为相对于y轴倾斜的角度：        
            计算出来的是radian弧度，通过 radian / np.pi * 180 得出的是相对于x轴倾斜(逆时针旋转)的角度，
            但我们的目的是计算相对于y轴倾斜的角度，因此通过((radian / np.pi * 180) - 90) 即基础上减去90度，
            那么角度变成从相对于x轴倾斜变成相对于y轴倾斜，最后通过((radian / np.pi * 180) - 90) * i 计算出相对于y轴倾斜的角度。
        角度转弧度：π/180×角度
        弧度变角度：180/π×弧度
        """
        # 旋转口罩：np.arctan2 计算出弧度
        radian = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
        # print("radian",radian) #比如弧度为 1.473543128543331
        # print("两点差值",chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])  #287  -109

        """ 
        1.chin_bottom_point：脸颊chin中的第9个特征点(下巴底部位置)
          nose_point：鼻梁nose_bridge中的第2个特征点
          左上角的坐标为(0,0)，脸颊chin中的第9个特征点和鼻梁nose_bridge中的第2个特征点构成一条直线line。
                鼻梁特征点的x 等于 下巴特征点的x，代表两点都位于y轴上，直线line和y轴重合。
                鼻梁特征点的x 大于 下巴特征点的x，直线line在y轴的左边，直线line相对于y轴倾斜 负数值的角度数。
                鼻梁特征点的x 小于 下巴特征点的x，直线line在y轴的右边，直线line相对于y轴倾斜 正数值的角度数。
        """
        angle = ((radian / np.pi * 180) - 90) * -1
        # print("angle",angle)

        """
        API官网 rotate解析：https://pillow.readthedocs.io/en/stable/reference/Image.html
        rotate(self, angle, resample=NEAREST, expand=0, center=None, translate=None, fillcolor=None,)
            返回一个按照给定角度顺时钟围绕图像中心旋转后的图像拷贝。
            expand=0：如果为false/0或者缺省，则使输出图像的大小与输入图像的大小相同。 请注意，展开标记假定围绕中心旋转且没有平移。
            expand=1：如果为true/1，则扩展输出图像以使其足够大以容纳整个旋转的图像。
 
        expand=True时，从 输入图像的尺寸(548, 307) 变成 输出图像的尺寸(622, 483)。
        expand=False(缺省值)时，输出图像与输入图像尺寸一样大，均为(548, 307)。
        """
        rotated_mask_img = mask_img.rotate(angle, expand=True)
        print("rotated_mask_img.size",rotated_mask_img.size) #(622, 483)
        print("mask_img.size",mask_img.size) #(548, 307)

        # rotated_mask_img = mask_img.rotate(angle)
        # print("rotated_mask_img.size",rotated_mask_img.size) #(548, 307)
        # print("mask_img.size",mask_img.size) #(548, 307)

        # 计算口罩位置
        #计算 鼻梁nose_bridge中的第2个特征点的x 与 脸颊chin中的第9个特征点(下巴底部位置)的x 的平均值
        center_x = (nose_point[0] + chin_bottom_point[0]) // 2
        print("center_x",center_x) # 比如 833
        #计算 鼻梁nose_bridge中的第2个特征点的y 与 脸颊chin中的第9个特征点(下巴底部位置)的y 的平均值
        center_y = (nose_point[1] + chin_bottom_point[1]) // 2
        print("center_y",center_y) # 比如 486

        # 口罩的宽// 2 - 左半边口罩的宽width：即一半口罩的宽 减去 左脸半边口罩的宽，差值如果为正值代表 左脸半边口罩的宽 小于 右脸半边口罩的宽，
        # 差值如果为负值代表 左脸半边口罩的宽 大于 右脸半边口罩的宽，不管是正值还是负值，
        # 差值实际就是 左脸半边口罩的宽 和 右脸半边口罩的宽 之间的差值。
        # offset = mask_img.width // 2 - mask_left_img.width
        # print("offset",offset) # 比如 -20，负值代表 左半边口罩的宽 大于 右半边口罩的宽。
        # print("mask_img.width",mask_img.width) #548。那么有 548 // 2 = 274 即一半口罩的宽
        # print("rotated_mask_img.width",rotated_mask_img.width) #622
        # print("mask_left_img.width",mask_left_img.width) #294 即 左脸半边口罩的宽，那么有 左脸半边口罩的宽 大于 右脸半边口罩的宽

        """
        1.(center_x,center_y)：
            鼻梁nose_bridge中的第2个特征点坐标位置 与 脸颊chin中的第9个特征点(下巴底部位置)坐标位置 的平均值，
            (center_x,center_y) 便大概落在鼻尖与上嘴唇之间的位置。
        2.rotated_mask_img.width // 2：旋转后的口罩的宽的一半
          rotated_mask_img.height // 2：旋转后的口罩的高的一半
        3.RAD表示弧度，Cos(1.47 rad)= Cos(84°13'29")= 0.100625733386932    
          角度转弧度：π/180×角度
          弧度变角度：180/π×弧度
        """
        # box_x = center_x + int(offset * np.cos(angle)) - rotated_mask_img.width // 2
        box_x = center_x - rotated_mask_img.width // 2
        # print("int(offset * np.cos(radian))",int(offset * np.cos(radian))) # 7
        # print("rotated_mask_img.width // 2",rotated_mask_img.width // 2) # 311

        # box_y = center_y + int(offset * np.sin(angle)) - rotated_mask_img.height // 2
        box_y = center_y - rotated_mask_img.height // 2
        # print("int(offset * np.sin(radian))",int(offset * np.sin(radian))) # -18
        # print("rotated_mask_img.height // 2",rotated_mask_img.height // 2) # 241

        # 往人脸上添加口罩
        self._face_img.paste(rotated_mask_img, (box_x,box_y), rotated_mask_img)

        from PIL import ImageDraw
        d = ImageDraw.Draw(self._face_img)
        d.point([(center_x,center_y)], fill=(0,0,255))
        d.point([chin_bottom_point, nose_point], fill=(255,255,255))
        d.line([chin_bottom_point, nose_point], fill=(255,0,0), width=2)
        d.line([nose_point, (nose_point[0], chin_bottom_point[1])], fill=(0,0,255), width=2)

    def _save(self):
        # os.path.splitext(“文件路径”)    分离文件名与扩展名；默认返回(fname,fextension)元组，可做分片操作
        path_splits = os.path.splitext(self.face_path)
        new_face_path = path_splits[0] + '-with-mask' + path_splits[1]
        self._face_img.save(new_face_path)
        print(f'Save to {new_face_path}')

    """
    1.point：
        1.在计算左半边脸上的口罩时，point为第1个特征点(x,y)：chin_left_point，即脸颊chin中的 第1个特征点的值，该特征点位于整个人脸68个特征点中的第1个
        2.在计算右半边脸上的口罩时，point为第17个特征点(x,y)：chin_right_point，即脸颊chin中的 第17个特征点的值，该特征点位于整个人脸68个特征点中的第17个
      line_point1：
        第9个特征点(x2,y2)：chin_bottom_point，即脸颊chin中的 第9个特征点的值，该特征点位于整个人脸68个特征点中的第9个，即下巴底部位置
      line_point2：
        第29个特征点(x1,y1)：nose_point，即鼻梁nose_bridge中的第2个特征点的值，该特征点位于整个人脸68个特征点中的第29个
    2.点(x0,y0) 到直线 ax+by+c=0 的距离d 的表达式为 |ax0+by0+c| / sqrt(a^2+b^2)
    3.根据已知两点坐标(两点式)的直线表达式为 (x-x1)/(x2-x1) = (y-y1)/(y2-y1)
            可以得知 点(x1,y1)到点(x2,y2) 构成一条直线，那么 点(x,y)为 点(x1,y1)到点(x2,y2) 这条直线上的点，或者说是在这条直线方向上的点，
            因为这条直线可以是所在的方向上的无限延伸的，所以可以说点(x,y) 为 点(x1,y1)到点(x2,y2) 这条直线(方向)上的点。
       (x-x1)/(x2-x1) = (y-y1)/(y2-y1) 表达式 简化为 x(y2-y1)+y(x1-x2)+y1(x2-x1)+x1(y1-y2)=0
       那么可以根据上式得知 点(x,y)中的x代表x0，y代表y0。a代表(y2-y1)，b代表(x1-x2)，c代表y1(x2-x1)+x1(y1-y2)。
       最终得 |ax0+by0+c| / sqrt(a^2+b^2) 
                = |(y2-y1)x + (x1-x2)y + y1(x2-x1)+x1(y1-y2)| / sqrt((y2-y1)^2 + (x1-x2)^2)
                = (x,y) 即为 point[0]和point[1]，也即 整个人脸68个特征点中的 第1个特征点(x,y) 或 第17个特征点(x,y)
                  (x1,y1) 即为 line_point1[0]和line_point1[1]，也即 整个人脸68个特征点中的 第9个特征点(x,y)
                  (x2,y2) 即为 line_point2[0]和line_point2[1]，也即 整个人脸68个特征点中的 第29个特征点(x,y)
                = |(line_point2[1]-line_point1[1])*point[0] + (line_point1[0]-line_point2[0])*point[1] 
                    + line_point1[1]*(line_point2[0]-line_point1[0]) + line_point1[0]*(line_point1[1]-line_point2[1])| 
                    / sqrt((line_point2[1]-line_point1[1])^2 + (line_point1[0]-line_point2[0])^2)
    """
    @staticmethod
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                          (line_point1[0] - line_point2[0]) * point[1] +
                          (line_point2[0] - line_point1[0]) * line_point1[1] +
                          (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                           (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)


# if __name__ == '__main__':
#     #cli()
#     create_mask(image_path)
