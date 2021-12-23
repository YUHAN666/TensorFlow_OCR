# coding=utf-

MODE = {0: "train_db",    # 训练模型分割部分
        1: "test_checkpoint",         # 验证模型分割效果
        2: "savePb",                # 将模型保存为pb文件
        3: "inspect_checkpoint",                # 查看checkpoint
        }

DefaultParam = {
	"mode": MODE[2],
    "anew": False,  # 是否需要新建模型，False则新建模型，True则从checkpoint中读取参数

    # Model param
    "momentum": 0.9,                # BatchNorm层动量参数
    "backbone": "res50",         # 骨架网络选择
    "neck": "dbnet",
    "decision_head": "ghost",

    # DataManager param
	"data_dir": "./dataset/pzt_carrier/",
	"image_size": 160,  # 320
	# "original_size": (1536, 2048),  #chip
	"original_size": (1200, 1920),  #chip
    "image_channel": 3,
    "augmentation_method": {"Fliplr": 0.5,
                            "Affine": (-0, 0),
                            "Resize": (0.9, 1.1),
                            "GaussianBlur": (0.0, 1.0)},


    # Trainer param
	"learning_rate": 3e-3,  # 学习率
    "optimizer": "Adam",                # 优化器选择  Adam GD RMS
    "save_frequency": 1,                # 保存checkpoint的频率，训练时每隔多少epoch保存一次模型
    "check_seg_frequency": 50,          # 多少epochs使用TensorBoard检查一次分割效果
    "max_to_keep": 3,                  # 最多保留的checkpoint文件个数
	"epochs": 500,  # 训练循环数
    "steps_per_epoch": 100,
	"batch_size": 4,  # 训练batch大小
	"batch_size_inference": 1,  # 保存为pb时batch大小
    "lr_decay": "exponential_decay",  # 学习率衰减策略   exponential_decay,inverse_time_decay,natural_exp_decay,cosine_decay
    "stair_case": False,                # 阶梯式衰减
	"decay_rate": 0.1,  # 衰减率，1则不衰减
	"decay_steps": 3000,  # 衰减步数
    "loss": "cross_entropy",            # 损失函数
	"warm_up": False,  # 预热学习率(先使用较小学习率，warm_up_step后增大至初始学习率以避免nan
	"warm_up_step": 3,  # 预热步数

    # Saver param
    "input_list": ["image_input"],        # pb模型输入node
    "output_list": ["dbnet/proba3_sigmoid"],      # pb模型输出节点
    "pb_save_path": "./pbMode/",    # pb模型路径
	"pb_name": "pzt_db_model.pb",  # pb模型命名
    "saving_mode": "regular",           # pb模型存储模式
    "checkpoint_dir_db": "./checkpoint/db/",        # checkpoint保存路径

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
