import time
import tensorflow as tf
from tqdm import tqdm
from db.db_losses import dice_loss, l1_loss, balanced_crossentropy_loss
from base.trainer import Trainer


class TrainerDB(Trainer):

    def __init__(self, sess, model, param, logger):
        self.session = sess
        self.model = model
        self.learning_rate = param["learning_rate"]
        self.optimizer = param["optimizer"]
        self.epochs = param["epochs"]
        self.steps_per_epoch = param["steps_per_epoch"]
        self.save_frequency = param["save_frequency"]
        self.mode = param["mode"]
        self.anew = param["anew"]
        self.loss = param["loss"]
        self.warm_up = param["warm_up"]
        self.warm_up_step = param["warm_up_step"]
        self.lr_decay = param["lr_decay"]
        self.decay_rate = param["decay_rate"]
        self.decay_steps = param["decay_steps"]
        self.staircase = param["stair_case"]
        self.check_seg_frequency = param["check_seg_frequency"]
        self.logger = logger

        with self.session.as_default():

            if self.lr_decay:
                self.global_step = tf.Variable(1, trainable=False)
                self.add_global = self.global_step.assign_add(1)
                self.learning_rate = self.learning_rate_decay()
            self.summary_learning_rate = tf.summary.scalar("learning_rate", self.learning_rate)

            if self.mode == "train_db":
                train_var_list = [v for v in tf.trainable_variables()]
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                # update_ops.append(tf.get_default_graph().get_operation_by_name("balanced_cross_entropy_cond"))
                optimizer = self.optimizer_func()
                loss, negative_count = self.loss_func(self.model.inputs, self.model.outputs)
                with tf.control_dependencies(update_ops):
                    optimize = optimizer.minimize(loss, var_list=train_var_list)
                self.loss = loss
                # self.positive_count = positive_count
                self.negative_count = negative_count
                self.optimize = optimize
                self.summary_loss_train = tf.summary.scalar("loss_train", self.loss)
                self.summary_loss_valid = tf.summary.scalar("loss_valid", self.loss)

    def optimizer_func(self):

        if self.optimizer == "Adam":
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optimizer == 'GD':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.optimizer == 'RMS':
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        else:
            raise ValueError("Unsupported optimizer {}".format(self.optimizer))

        return optimizer

    def learning_rate_decay(self):

        if self.lr_decay == "exponential_decay":
            # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
            decayed_learning_rate = tf.train.exponential_decay(self.learning_rate, global_step=self.global_step,
                                                               decay_steps=self.decay_steps, decay_rate=self.decay_rate,
                                                               staircase=self.staircase)
        elif self.lr_decay == "inverse_time_decay":
            # decayed_learning_rate = learning_rate / (1 + decay_rate * global_step / decay_step)
            decayed_learning_rate = tf.train.inverse_time_decay(self.learning_rate, global_step=self.global_step,
                                                                decay_steps=self.decay_steps,
                                                                decay_rate=self.decay_rate,
                                                                staircase=self.staircase)
        elif self.lr_decay == "natural_exp_decay":
            # decayed_learning_rate = learning_rate * exp(-decay_rate * global_step / decay_steps)
            decayed_learning_rate = tf.train.natural_exp_decay(self.learning_rate, global_step=self.global_step,
                                                               decay_steps=self.decay_steps, decay_rate=self.decay_rate,
                                                               staircase=self.staircase)
        elif self.lr_decay == "cosine_decay":
            # cosine_decay = 0.5 * (1 + cos(pi * global_step / decay_steps))
            # decayed = (1 - alpha) * cosine_decay + alpha
            # decayed_learning_rate = learning_rate * decayed
            # alpha的作用可以看作是baseline，保证lr不会低于某个值。不同alpha的影响如下：
            decayed_learning_rate = tf.train.cosine_decay(self.learning_rate, global_step=self.global_step,
                                                          decay_steps=self.decay_steps, alpha=0.3)
        else:
            raise ValueError("Unsupported learning rate decay strategy {}".format(self.lr_decay))

        if self.warm_up:
            warmup_learn_rate = self.learning_rate * tf.cast(self.global_step / self.warm_up_step, tf.float32)
            learning_rate = tf.cond(self.global_step <= self.warm_up_step, lambda: warmup_learn_rate,
                                    lambda: decayed_learning_rate)
        else:
            learning_rate = decayed_learning_rate
        return learning_rate

    def loss_func(self, ground_truth, prediction):

        gt, mask, thresh_map, thresh_mask = ground_truth
        binary, thresh, thresh_binary = prediction

        l1_loss_ = l1_loss([thresh, thresh_map, thresh_mask])
        balanced_ce_loss_, dice_loss_weights, negative_count = balanced_crossentropy_loss([binary, gt, mask])
        dice_loss_ = dice_loss([thresh_binary, gt, mask, dice_loss_weights])

        return l1_loss_ + balanced_ce_loss_ + dice_loss_, negative_count

    def train(self, data_manager, saver):
        """ Train the segmentation part of the model """
        self.logger.info("Start training segmentation for {} epochs, {} steps per epochs, batch size is {}. "
                         "Save to checkpoint every {} epochs "
                         .format(self.epochs, data_manager.num_batch_train, data_manager.batch_size, self.save_frequency))
        if self.lr_decay:
            lr = self.session.run([self.learning_rate])
        else:
            lr = self.learning_rate
        self.logger.info("Loss: {}, Optimizer: {}, Learning_rate: {}".format(self.loss, self.optimizer, lr))
        if self.lr_decay:
            self.logger.info("Using {} strategy, decay_rate: {}， decay_steps: {}, staircase: {}".format(self.lr_decay, self.decay_rate, self.decay_steps, self.staircase))
        current_epoch = saver.step + 1
        with self.session.as_default():
            print('Start training segmentation for {} epochs, {} steps per epoch'.format(self.epochs, data_manager.num_batch_train))
            tensorboard_merged = tf.summary.merge([self.summary_learning_rate, self.summary_loss_train])
            train_loss = [] # 记录每个epoch的loss
            val_loss = []


            for i in range(current_epoch, self.epochs+current_epoch):
                print('Epoch {}:'.format(i))
                iter_loss = []  # 记录每个step的loss
                time.sleep(0.1)
                pbar = tqdm(total=data_manager.num_batch_train, leave=True)
                # epoch start
                for batch in range(data_manager.num_batch_train):
                    # batch start

                    image_batch, gt_batch, mask_batch, thresh_batch, thresh_mask_batch = self.session.run(data_manager.next_batch_train)

                    _, loss_value_batch, tensorboard_result, negative_count, learning_rate = self.session.run([self.optimize,
                                                                                self.loss,
                                                                                tensorboard_merged, self.negative_count,self.learning_rate],
                                                              feed_dict={self.model.image_input: image_batch,
                                                                         self.model.gt_input: gt_batch,
                                                                         self.model.mask_input: mask_batch,
                                                                         self.model.thresh_input: thresh_batch,
                                                                         self.model.thresh_mask_input: thresh_mask_batch,
                                                                         self.model.is_training: True})

                    # print('negative_count: {}'.format(negative_count))
                    iter_loss.append(loss_value_batch)
                    pbar.update(1)
                    if self.lr_decay:
                        _, lr = self.session.run([self.add_global, self.learning_rate])
                    # print('positive count: {}       negative count: {}'.format(positive_count, negative_count))
                print('learning rate: {}'.format(learning_rate))
                pbar.clear()
                pbar.close()
                time.sleep(0.1)
                # loss and iou check
                train_loss.append(sum(iter_loss)/len(iter_loss))
                # val_loss_epo = self.validation(data_manager_valid, i)
                # val_loss.append(val_loss_epo)
                # self.logger.info("Epoch{}  train_loss:{},  val_loss:{}"
                #                  .format(i, iter_loss[i-current_epoch], val_loss[i-current_epoch]))
                # print('train_loss:{}, val_loss:{}'
                #       .format(iter_loss[i-current_epoch], val_loss[i-current_epoch]))
                print('train_loss:{}'
                      .format(iter_loss[i-current_epoch]))

                if (i-current_epoch+1) % self.save_frequency == 0 or i == self.epochs + current_epoch:

                    saver.save_checkpoint(i)

    def valid(self, data_manager):
        """ Evaluate the segmentation part during training process"""
        with self.session.as_default():
            # print('start validating segmentation')
            total_loss = 0.0
            num_step = 0.0
            for batch in range(data_manager.num_batch_test):
                image_batch, gt_batch, mask_batch, thresh_batch, thresh_mask_batch = self.session.run(data_manager.next_batch_test)

                total_loss_value_batch, tensorboard_result = self.session.run([self.loss,
                                                                               self.summary_loss_valid],
                                                             feed_dict={self.model.image_input: image_batch,
                                                                         self.model.gt_input: gt_batch,
                                                                         self.model.mask_input: mask_batch,
                                                                         self.model.thresh_input: thresh_batch,
                                                                         self.model.thresh_mask_input: thresh_mask_batch,
                                                                         self.model.is_training: False})
                num_step = num_step + 1
                total_loss += total_loss_value_batch

            total_loss = total_loss/num_step
            return total_loss




