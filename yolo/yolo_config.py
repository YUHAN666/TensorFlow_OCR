# coding=utf-8

MODE = {0: "train_yolo",    # 训练模型分割部分
        1: "inference",
        2: "savePb",                # 将模型保存为pb文件
        3: "inspect_checkpoint",                # 查看checkpoint
        }
# LABEL_DICT = {'OCR': 0}
# LABEL_DICT = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}
# LABEL_DICT = {"1": 0, "2": 1}
LABEL_DICT = {"pzt":0,}
DefaultParam = {
	"mode": MODE[1],
	"anew": False,  # 是否需要新建模型，False则新建模型，True则从checkpoint中读取参数

    # Model param
    "momentum": 0.9,                # BatchNorm层动量参数
    "backbone": "darknet",         # 骨架网络选择
    "neck": "yolo",
    "decision_head": "yolo_head",

	"learning_rate": 7e-5,  # 学习率
    "batch_size": 4,  # 训练batch大小

    # DataManager param
	"data_dir": "./dataset/yolo_pzt/",  # 按照VOC格式制作的数据集，目录下一定要有JPEGImages和Annotations两个文件夹
    "txt_path": "./yolo/data_list.txt",                               # 首次读取数据集后，将图片路径和box信息写入指定txt文件，之后使用可直接读取
    "anchor_path": "./yolo/yolo_anchors.txt",                   # 首次训练时根据数据集的anchor大小kmeans求出clustter_num个anchors然后将anhors的width,height写入指定txt文件
	"clustter_num": 4,  # anchor数，等于9时使用darknet yolo model；等于6时使用tiny yolo model
    "num_classes": len(LABEL_DICT.keys()),
	"down_scale": 16,
	# "image_size": (1536, 2048),                                 # 输入图片尺寸（此项目为固定尺寸输入）在infer时用到
    # "image_size": (1200, 1920),
	# "image_size": (1200, 1920),
	"image_size": (1200, 1920),
	"input_shape": (320, 320),  # 模型输入尺寸
	"image_channel": 3,  # only used in customized model
	"inference_dir": './dataset/yolo_pzt/JPEGImages-test/',
    "augmentation_method": {"Fliplr": 0.5,
                            "Rotate": (0, 0),
                            "Resize": .1,
                            "GaussianBlur": (0.0, 1.1)},
    "mosaic": False,                     # 是否使用mosaic数据增强
    "iou_loss": False,                   # 使用IOU LOSS或者MSE loss回归box的 x,y,w,h

    # Trainer param
	"ignore_thres": 0.7,
	# threshold for ignoring negative confidence loss during training, 忽略小于此thresh的box confidence loss 训练时先大后小
	"score_thres": 0.2,  # threshold for filtering boxes during inference
	"iou_thres": 0.2,  # threshold for NMS during inference 越低则boxes重叠的越少
	"max_boxes": 7,  # max number of boxes to detect during inference
    "optimizer": "Adam",                # 优化器选择  Adam GD RMS
    "save_frequency": 1,                # 保存checkpoint的频率，训练时每隔多少epoch保存一次模型
    "check_seg_frequency": 50,          # 多少epochs使用TensorBoard检查一次分割效果
    "max_to_keep": 3,                   # 最多保留的checkpoint文件个数
    "epochs": 300,                      # 训练循环数
    "steps_per_epoch": 100,
    "batch_size_inference": 1,          # 保存为pb时batch大小
    "lr_decay": "exponential_decay",  # 学习率衰减策略   exponential_decay,inverse_time_decay,natural_exp_decay,cosine_decay
    "stair_case": False,                # 阶梯式衰减
    "decay_rate": 0.1,                  # 衰减率，1则不衰减
	"decay_steps": 1000,  # 衰减步数
    "loss": "cross_entropy",            # 损失函数
	"warm_up": False,  # 预热学习率(先使用较小学习率，warm_up_step后增大至初始学习率以避免nan
    "warm_up_step": 500,  # 预热步数

    # Saver param
	"input_list": ["yolo_image_input", "yolo_input_shape", "yolo_image_shape"],  # pb模型输入node
	"output_list": ["yolo_output_boxes", "yolo_output_scores", "yolo_output_classes"],  # pb模型输出节点
    "pb_save_path": "./pbMode/",    # pb模型路径
    "pb_name": "123.pb",            # pb模型命名
    "saving_mode": "regular",           # pb模型存储模式
	"checkpoint_dir": "./checkpoint/yolo/",  # checkpoint保存路径

    # Logger & TensorBoard param
    "log_path": "./Log/",            # Log文件保存路径
    "tensorboard_dir": "./tensorboard/",  # TensorBoard event文件输出路径
    "need_clear_tensorboard": True      # 是否需要清空TensorBoard输出目录下的文件
}

BIN_SIZE = [1, 2, 4, 7]
ACTIVATION = 'relu'  # swish mish or relu
ATTENTION = 'se'  # se or cbam
DATA_FORMAT = 'channels_last'
DROP_OUT = False
TEST_RATIO = 0.25
CLASS_NUM = 1


