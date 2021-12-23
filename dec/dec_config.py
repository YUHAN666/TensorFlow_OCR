# coding=utf-8

BIN_SIZE = [1, 2, 4, 7]
ACTIVATION = 'relu'  # swish mish or relu
ATTENTION = 'se'  # se or cbam
DATA_FORMAT = 'channels_last'
DROP_OUT = False

MODE = {0: "train_dec",        # 训练模型分类部分
        1: "test",         # 验证模型分割效果
        2: "savePb",                # 将模型保存为pb文件
        3: "simple_save"
        }

DefaultParam = {
    "mode": MODE[2],
    "anew": False,  # 是否需要新建模型，False则新建模型，True则从checkpoint中读取参数

    # Model param
    "momentum": 0.9,                # BatchNorm层动量参数
    "backbone": "ghost",         # 骨架网络选择

    # DataManager param
	"data_dir": "./dataset/pzt_carrier_dec/",
    "cut_image_channel": 3,
	"cut_image_size": 96,  # 96
    "augmentation_method": {"GammaContrast": (0.8, 1.2),
                            "Affine": (-10, 10),
                            "MotionBlur": ((3, 7), 90, (0, 0))},

    # Trainer param
	"learning_rate": 3e-4,  # 学习率
    "optimizer": "Adam",                # 优化器选择  Adam GD RMS
    "save_frequency": 1,                # 保存checkpoint的频率，训练时每隔多少epoch保存一次模型
    "check_seg_frequency": 50,          # 多少epochs使用TensorBoard检查一次分割效果
    "max_to_keep": 3,                  # 最多保留的checkpoint文件个数
    "epochs": 30,                       # 训练循环数
    "steps_per_epoch": 100,
    "batch_size_cut": 64,
    "batch_size_inference": 1,          # 保存为pb时batch大小
    "lr_decay": "exponential_decay",  # 学习率衰减策略   exponential_decay,inverse_time_decay,natural_exp_decay,cosine_decay
    "stair_case": False,                # 阶梯式衰减
    "decay_rate": 0.1,                  # 衰减率，1则不衰减
	"decay_steps": 1000,  # 衰减步数
    "loss": "cross_entropy",            # 损失函数
    "warm_up": True,  # 预热学习率(先使用较小学习率，warm_up_step后增大至初始学习率以避免nan
	"warm_up_step": 50,  # 预热步数

    # Saver param
    "input_list": ["cut_image_input"],        # pb模型输入node
    "output_list": ["decision_out", "decision_prob"],      # pb模型输出节点
    "pb_save_path": "./pbMode/",    # pb模型路径
	"pb_name": "pzt_dec_model.pb",  # pb模型命名
    "saving_mode": "CBR",           # pb模型存储模式
    "checkpoint_dir_dec": "./checkpoint/dec/",

    # Logger & TensorBoard param
    "log_path": "./Log/",            # Log文件保存路径
    "tensorboard_dir": "./tensorboard/",  # TensorBoard event文件输出路径
    "need_clear_tensorboard": True      # 是否需要清空TensorBoard输出目录下的文件
}

