# -*- coding: utf-8 -*-
# @Author  : lzw
# description : logistic regression 实现，包括正则化参数l1, l2，梯度下降
# requirements : python3
import logging
import numpy as np
# from scipy.optimize import fmin_bfgs
# import bigfloat
logging.getLogger().setLevel(logging.INFO)


class lr_model:
    def __init__(self,
                 batch_size=10,
                 llambda=0.1,
                 epsilon=1e-4,
                 alpha=0.01,
                 tolerance=5,
                 num_iters=5000,
                 debug=False,
                 normalization=None):
        '''
        :param data:
        :param labels:
        :param alpha:             学习率
        :param num_iters:         训练迭代次数，
        :param debug:
        :param normalization:     正则化类型
        '''
        self.normalization_mode = normalization
        self.debug = debug
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.tolerance = tolerance
        self.alpha = alpha
        self.llambda = llambda
        self.epsilon = epsilon
        self.batch_size = batch_size

    def logSigmoidCalc(self, Z):
        '''
        计算logsigmoid
        这里进来的Z稍微大一点，计算sigmoid就变成 除以一个很大的数，
        使得返回的sigmoid值为0，间接造成 divide by zero.
        所以为了避免，初始化值theta要小一点
        theta 小一点但是还有问题。。。
        :param Z:
        :return:
        '''
        if Z.min() < 0:
            g = Z - np.log(1 + np.exp(Z))
        else:
            g = - np.log(1 + np.exp(-Z))
        return g

    def sigmoidCalc(self, Z):
        '''
        计算sigmoid
        这里进来的Z稍微大一点，计算sigmoid就变成 除以一个很大的数，
        使得返回的sigmoid值为0，间接造成 divide by zero.
        所以为了避免，初始化值theta要小一点
        theta 小一点但是还有问题。。。
        :param Z:
        :return:
        '''
        if Z.min() < 0:
            g = np.exp(Z) / (1 + np.exp(Z))
        else:
            g = 1.0 / (1 + np.exp(-Z))
        return g

    def train(self, data, labels, unique_classes):
        '''
        训练分类器；考虑多分类情况
        '''
        self.m, self.n = data.shape  # m笔数据；且有n个特征
        assert (len(np.unique(labels)) >= 2)

        num_classes = len(unique_classes)  # 总共有几个类别
        Init_Thetas = []  # theta 的初始化值们
        self.Thetas = []  # thera 的最终值们
        self.Costs = []  # 根据Theta算出来的损失函数值

        if (num_classes == 2):
            # 我们需要一个分类器来分类 class A, class B
            theta_init = np.random.normal(loc=0, scale=0.01, size=(self.n, 1))  # 每个特征一个权重值 theta
            # theta_init = np.zeros((self.n, 1))
            Init_Thetas.append(theta_init)
            local_labels = labels

            logging.info('[LR] start training a lr model with 0/1 labels')
            init_theta = Init_Thetas[0]
            new_theta, final_cost = self.computeGradient(data, local_labels, init_theta)
            self.Thetas.append(new_theta)
            self.Costs.append(final_cost)

        elif (num_classes > 2):
            '''
            # 对于多分类模型，这里使用 one vs all 的方式
            # 也就是把其中的一类当成正类，其余的样本都是负类
            # 对于m个类别，一共得训练 m 个分类器
            if num_classes = 2, then N_Thetas will contain only 1 Theta
            if num_classes >2, then N_Thetas will contain num_classes number of Thetas.
            '''
            for eachInitTheta in range(num_classes):
                theta_init = np.random.normal(loc=0, scale=0.01, size=(self.n, 1))  # 每个特征一个权重值 theta
                # theta_init = np.zeros((self.n,1))
                Init_Thetas.append(theta_init)
                pass

            for eachClass in range(num_classes):
                local_labels = np.zeros(labels.shape)
                local_labels[np.where(labels == eachClass)] = 1
                assert len(np.unique(local_labels)) == 2
                assert len(local_labels) == len(labels)

                logging.info('[LR] start training a lr model with multiple labels')
                init_theta = Init_Thetas[eachClass]
                new_theta, final_cost = self.computeGradient(data, local_labels, init_theta)
                self.Thetas.append(new_theta)
                self.Costs.append(final_cost)

        return self.Thetas, self.Costs

    def computeCost(self, data, labels, theta):
        '''
        compute cost of the given value of theta and return it
        根据当前数据的和权重值theta计算损失函数值
        '''

        theta2 = theta[range(1, theta.shape[0]), :]  # 提取出权重的部分，除了 b
        regularized_parameter = 0
        if self.normalization_mode == "l1" and self.llambda > 0:
            regularized_parameter = np.dot(self.llambda / (2 * self.m), np.sum(np.abs(theta2)))
        elif self.normalization_mode == "l2" and self.llambda > 0:
            regularized_parameter = np.dot(self.llambda / (2 * self.m), np.sum(theta2 * theta2))
        elif self.normalization_mode != "l1" and self.normalization_mode != "l2" and self.llambda > 0:
            logging.info('[LR] 错误指定正则化类型')
            exit()

        Z = np.dot(data, theta)
        # J = -1.0 * np.sum(
        #             np.log(self.sigmoidCalc(Z) * labels + 1 - self.sigmoidCalc(Z) * (1 - labels))
        #              )
        J = (1.0 / self.m) * np.sum(-1.0 * labels * Z + np.log(1 + np.exp(Z)))
        J = J + regularized_parameter

        if np.isnan(J) or J == float('inf'):  # 当出现nan值时的纠错
            print('---' * 5)
            print(self.sigmoidCalc(Z) * labels)
            assert not np.isnan(J)
            assert J != float('inf')
        return J

    def computeGradient(self, data, labels, theta):
        """
        梯度下降
        BGD: 批量梯度下降
        SGD: 利用每个样本的损失函数对θ求偏导得到对应的梯度，来更新θ
        MBGD: 每次更新参数时使用b个样本（b一般为10）
        :param data:
        :param labels:
        :param init_theta:
        :return:
        """
        self.shuffle = True
        data_size = len(data)
        assert data_size == len(labels)
        data_labels = list(zip(data, labels))

        # 打乱数据，采用批量梯度下降法
        if self.shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = np.array(data_labels)[shuffle_indices]
        else:
            shuffled_data = data_labels

        # 如果想要用 mbgd, 小批量梯度下降法
        num_batches_per_epoch = int(data_size // self.batch_size + 1) if data_size % self.batch_size != 0 else int(
            data_size / self.batch_size)
        batches = []
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * self.batch_size
            end_index = min((batch_num + 1) * self.batch_size, data_size)
            # if len(shuffled_data[start_index:end_index]) != self.batch_size:
            #     continue
            batches.append(shuffled_data[start_index:end_index])
        batches_len = len(batches)

        last_loss, loss = float('inf'), float('inf')
        tolerance = self.tolerance
        step = 0
        for _ in range(self.num_iters):
            for _ in range(batches_len):
                x_train_batch, y_train_batch = zip(*shuffled_data)
                x_train_batch = np.array(x_train_batch)
                y_train_batch = np.array(y_train_batch).reshape(-1, 1)

                self.probas = self.sigmoidCalc(np.dot(x_train_batch, theta))
                error = self.probas - y_train_batch
                gw = (1 / self.batch_size) * np.dot(x_train_batch.T, error)
                g0 = gw[0]  # theta_0 和 regularization 无关

                # 对参数的更新加入正则化项
                if self.normalization_mode == "l1" and self.llambda > 0:
                    """
                    采用L1 regularizer，其优良性质是能产生稀疏性，导致  W 中许多项变成零。
                    稀疏解，除了计算量上的好处之外，更重要的是更具有“可解释性”。
                    """
                    gw += self.llambda / self.batch_size
                elif self.normalization_mode == "l2" and self.llambda > 0:
                    """
                    采用L2 regularizer，使得模型的解偏向于 norm 较小的 W，通过限制 W 的 norm 的大小，
                    实现了对模型空间的限制；不过 ridge regression 并不具有产生稀疏解的能力，得到的系数
                    仍然需要数据中的所有特征才能计算预测结果，从计算量上来说并没有得到改观。
                    """
                    gw += (self.llambda * theta) / self.batch_size

                gw[0] = g0  # theta_0 和 regularization 无关
                theta -= self.alpha * gw  # 更新参数

                # 计算 magnitude of the gradient 并确认是否收敛
                # loss = np.linalg.norm(gw)
                loss = self.computeCost(x_train_batch, y_train_batch, theta)
                if self.epsilon > last_loss - loss:
                    if tolerance > 0:
                        tolerance -= 1
                    else:
                        logging.info('[LR] --Early stop at Step {} , cost = {} . '.format(step, loss))
                        return theta, loss
                logging.info('[LR] --Step {} , cost = {} . '.format(step, loss))
                last_loss = loss
                step += 1
        return theta, loss

    def get_proba(self, X, theta):
        return 1.0 / (1 + np.exp(- np.dot(X, theta)))

    def classify(self, data):
        """
        :param data:
        :param Thetas: 权重
        :return:
        """
        assert (len(self.Thetas) > 0)
        if (len(self.Thetas) > 1):  # 多分类的情况
            mvals = []
            for eachTheta in self.Thetas:
                classification_val = self.sigmoidCalc(np.dot(data, eachTheta))  # 作为第 i 类的概率
                mvals.append(classification_val)

            mvals = np.array(mvals)
            return list(zip(np.max(mvals, 0), np.argmax(mvals, 0)))

        elif (len(self.Thetas) == 1):
            # 实现向量化vectorize，并输出 [0,1] 概率值
            cval = self.sigmoidCalc(np.dot(data, self.Thetas[0]))  # probability
            return list(zip(cval, np.around(cval)))


# 处理类别的映射
# labels = np.zeros(Olabels.shape)
# uniq_Olabel_names = np.unique(Olabels)
# uniq_label_list = range(len(uniq_Olabel_names))
#
# for each in zip(uniq_Olabel_names, uniq_label_list):
#     o_label_name = each[0]
#     new_label_name = each[1]
#     labels[np.where(Olabels == o_label_name)] = new_label_name
#
# labels = labels.reshape((len(labels), 1))
