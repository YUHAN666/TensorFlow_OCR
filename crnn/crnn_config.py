# coding=utf-8

MODE = {0: "train",  # 训练模型分割部分
        1: "test",
        2: "savePb",  # 将模型保存为pb文件
        3: "inspect_checkpoint",  # 查看checkpoint
        }
# LABEL_DICT = {'OCR': 0}
LABEL_DICT = {'a': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '0': 10, '-': 11}
DefaultParam = {
	"mode": MODE[2],
	"anew": False,  # 是否需要新建模型，False则新建模型，True则从checkpoint中读取参数

	# Model param
	"momentum": 0.9,  # BatchNorm层动量参数
	"learning_rate": 3e-6,  # 学习率
	"batch_size": 32,  # 训练batch大小

	# DataManager param
	"data_list_path": "E:/CODES/TensorFlow_OCR/crnn/data_list.txt",
	"train_list_path": "E:/CODES/TensorFlow_OCR/crnn/train_list.txt",
	"test_list_path": "E:/CODES/TensorFlow_OCR/crnn/test_list.txt",

	"class_num": len(LABEL_DICT.keys()) + 1,
	"image_height": 32,
	"image_width": 100,
	"image_channel": 3,  # only used in customized model
	"max_text_len": 5,
	"hidden_units": 256,
	'hidden_layers': 2,

	"inference_dir": './dataset/yolo_number/test_images/',
	"augmentation_method": {"Fliplr": 0.5,
	                        "Rotate": (0, 0),
	                        "Resize": .1,
	                        "GaussianBlur": (0.0, 1.1)},
	# Trainer param

	"optimizer": "Adadelta",  # 优化器选择  Adam GD RMS
	"save_frequency": 1,  # 保存checkpoint的频率，训练时每隔多少epoch保存一次模型
	"check_seg_frequency": 50,  # 多少epochs使用TensorBoard检查一次分割效果
	"max_to_keep": 3,  # 最多保留的checkpoint文件个数
	"epochs": 300,  # 训练循环数
	"steps_per_epoch": 100,
	"batch_size_inference": 1,  # 保存为pb时batch大小
	"lr_decay": "exponential_decay",  # 学习率衰减策略   exponential_decay,inverse_time_decay,natural_exp_decay,cosine_decay
	"stair_case": False,  # 阶梯式衰减
	"decay_rate": 0.1,  # 衰减率，1则不衰减
	"decay_steps": 800,  # 衰减步数
	"loss": "cross_entropy",  # 损失函数
	"warm_up": False,  # 预热学习率(先使用较小学习率，warm_up_step后增大至初始学习率以避免nan
	"warm_up_step": 500,  # 预热步数

	# Saver param
	"input_list": ["yolo_image_input", "yolo_input_shape", "yolo_image_shape"],  # pb模型输入node
	"output_list": ["yolo_output_boxes", "yolo_output_scores", "yolo_output_classes"],  # pb模型输出节点
	"pb_save_path": "./pbMode/",  # pb模型路径
	"pb_name": "123.pb",  # pb模型命名
	"saving_mode": "regular",  # pb模型存储模式
	"checkpoint_dir": "./checkpoint/crnn/",  # checkpoint保存路径

	# Logger & TensorBoard param
	"log_path": "./Log/",  # Log文件保存路径
	"tensorboard_dir": "./tensorboard/",  # TensorBoard event文件输出路径
	"need_clear_tensorboard": True  # 是否需要清空TensorBoard输出目录下的文件
}

BIN_SIZE = [1, 2, 4, 7]
ACTIVATION = 'relu'  # swish mish or relu
ATTENTION = 'se'  # se or cbam
DATA_FORMAT = 'channels_last'
DROP_OUT = False
TEST_RATIO = 0.25
CLASS_NUM = 1
