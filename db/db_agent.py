import tensorflow as tf

from base.agent import Agent
from db.db_data_manager import DataManagerDB
from db.db_model import ModelDB
from db.db_trainer import TrainerDB
from inference.checkpoint_tester import Validator
from saver import Saver


class AgentDB(Agent):

    def __init__(self, param, logger):

        self.logger = logger
        logger.info("Start initializing AgentDB, mode is {}".format(param["mode"]))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.param = param
        self.session = tf.Session(config=config)
        self.model = ModelDB(self.session, self.param, self.logger)    # 建立将模型graph并写入session中
        self.data_manager = DataManagerDB(self.param)
        self.trainer = TrainerDB(self.session, self.model, self.param, self.logger)    # 损失函数，优化器，以及训练策略
        self.saver = Saver(self.session, self.param, self.model.checkPoint_dir, self.logger, self.model)     # 用于将session保存到checkpoint或pb文件中
        logger.info("Successfully initialized")

    def run(self):

        if not self.param["anew"] and self.param["mode"] != "testPb":
            self.saver.load_checkpoint()
        if self.param["mode"] == "train_db":      # 训练模型分割部分
            self.trainer.train(self.data_manager, self.saver)
        elif self.param["mode"] == "savePb":                # 保存模型到pb文件
            self.saver.save_pb()
        elif self.param["mode"] == "test_checkpoint":
            self.chechpoint_tester = Validator(self.session, self.model, self.logger, self.param["original_size"])
            # self.chechpoint_tester.test_kmeans(data_manager=self.data_manager, image_size=(1920, 1200))
            self.chechpoint_tester.visualization(data_manager=self.data_manager)
        elif self.param["mode"] == "inspect_checkpoint":
            self.saver.inspect_checkpoint()


