#%%
import os
import datetime
import time
import torch
from torch.utils.data import DataLoader

from net import MyNet
from data import MyDataset
from my_arguments import Args
from main import CLASSES, GL_NUMBBOX, GL_NUMGRID
from util import labels2bbox



class TrainInterface(object):
    """
    网络训练接口，
    __train(): 训练过程函数
    __validate(): 验证过程函数
    __save_model(): 保存模型函数
    main(): 训练网络主函数
    """
    def __init__(self, opts):
        """
        :param opts: 命令行参数
        """
        self.opts = opts
        print("=======================Start training.=======================")

    @staticmethod
    def __train(model, train_loader, optimizer, epoch, num_train, opts):
        """
        完成一个epoch的训练
        :param model: torch.nn.Module, 需要训练的网络
        :param train_loader: torch.utils.data.Dataset, 训练数据集对应的类
        :param optimizer: torch.optim.Optimizer, 优化网络参数的优化器
        :param epoch: int, 表明当前训练的是第几个epoch
        :param num_train: int, 训练集数量
        :param opts: 命令行参数
        """
        model.train()
        device = opts.GPU_id
        avg_metric = 0.  # 平均评价指标
        avg_loss = 0.  # 平均损失数值
        # log_file是保存网络训练过程信息的文件，网络训练信息会以追加的形式打印在log.txt里，不会覆盖原有log文件
        log_file = open(os.path.join(opts.checkpoints_dir, "log.txt"), "a+")
        localtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 打印训练时间
        log_file.write(localtime)
        log_file.write("\n======================training epoch %d======================\n"%epoch)
        for i,(imgs, labels) in enumerate(train_loader):
            labels = labels.view(opts.batch_size, GL_NUMGRID, GL_NUMGRID, -1)   #(batch_size,7,7,30)
            labels = labels.permute(0,3,1,2) #(batch_size,30,7,7)
            if opts.use_GPU:
                imgs = imgs.to(device)
                labels = labels.to(device)
            preds = model(imgs)  # 前向传播
            loss = model.calculate_loss(labels)  # 计算损失
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 优化网络参数
            # metric = model.calculate_metric(preds, labels)  # 计算评价指标
            # avg_metric = (avg_metric*i+metric)/(i+1)
            avg_loss = (avg_loss*i+loss.item())/(i+1)
            if i % opts.print_freq == 0:  # 根据打印频率输出log信息和训练信息
                print("Epoch %d/%d | Iter %d/%d | training loss = %.3f, avg_loss = %.3f" %
                      (epoch, opts.epoch, i, num_train//opts.batch_size, loss.item(), avg_loss))
                log_file.write("Epoch %d/%d | Iter %d/%d | training loss = %.3f, avg_loss = %.3f\n" %
                      (epoch, opts.epoch, i, num_train//opts.batch_size, loss.item(), avg_loss))
                log_file.flush()
        log_file.close()

    @staticmethod
    def __validate(model, val_loader, epoch, num_val, opts):
        """
        完成一个epoch训练后的验证任务
        :param model: torch.nn.Module, 需要训练的网络
        :param _loader: torch.utils.data.Dataset, 验证数据集对应的类
        :param epoch: int, 表明当前训练的是第几个epoch
        :param num_val: int, 验证集数量
        :param opts: 命令行参数
        """
        model.eval()
        log_file = open(os.path.join(opts.checkpoints_dir, "log.txt"), "a+")
        log_file.write("======================validate epoch %d======================\n"%epoch)
        log_file.flush()
        metrics=[]
        device=opts.GPU_id
        avg_metric = 0.
        """"
        label天生是 n,7,7,30的
        net的输出是 n,30,7,7的
        现在要做的是计算metric，要把他们拉平到(n,-1)的水平
        """
        with torch.no_grad():  # 加上这个可以减少在validation过程时的显存占用，提高代码的显存利用率
            for i,(imgs, labels) in enumerate(val_loader):   # labels.shape=(batch_size,1470)，由(batch_size,7,7,30)拉平得到
                labels = labels.view(opts.batch_size, GL_NUMGRID, GL_NUMGRID, -1)  # (batch_size,7,7,30)
                labels = labels.permute(0, 3, 1, 2)  # (batch_size,30,7,7)
                labels = labels.reshape(opts.batch_size,-1)
                if opts.use_GPU:
                    imgs = imgs.to(device)
                # net的输出为(batch_size  30  7  7)
                preds = model(imgs).cpu()
                batchsize=preds.shape[0]
                preds=preds.reshape(batchsize,-1)  #拉平，得到(batch_size,1470)
                metric=model.calculate_metric(preds, labels)
                metric=metric.item()       #转化为float值，并加入metrics中
                metrics.append(metric)
            avg_metric=sum(metrics)/len(metrics)    #求和，返回均值
            print("Evaluation of validation result: average L2 distance = %.5f"%(avg_metric))
            log_file.write("Evaluation of validation result: average L2 distance = %.5f\n"%(avg_metric))
            log_file.flush()
            log_file.close()
        return metric

    @staticmethod
    def __save_model(model, epoch, opts):
        """
        保存第epoch个网络的参数
        :param model: torch.nn.Module, 需要训练的网络
        :param epoch: int, 表明当前训练的是第几个epoch
        :param opts: 命令行参数
        """
        model_name = "epoch%d.pkl" % epoch
        save_dir = os.path.join(opts.checkpoints_dir, model_name)
        torch.save(model, save_dir)


    def main(self):
        """
        训练接口主函数，完成整个训练流程
        1. 创建训练集和验证集的DataLoader类
        2. 初始化带训练的网络
        3. 选择合适的优化器
        4. 训练并验证指定个epoch，保存其中评价指标最好的模型，并打印训练过程信息
        5. TODO: 可视化训练过程信息
        """
        opts = self.opts
        if not os.path.exists(opts.checkpoints_dir):
            os.mkdir(opts.checkpoints_dir)
        random_seed = opts.random_seed
        train_dataset = MyDataset(opts.dataset_dir, seed=random_seed, mode="train")
        val_dataset = MyDataset(opts.dataset_dir, seed=random_seed, mode="test")
        train_loader = DataLoader(train_dataset, opts.batch_size, shuffle=True, num_workers=opts.num_workers,drop_last=True)
        val_loader = DataLoader(val_dataset, opts.batch_size, shuffle=False, num_workers=opts.num_workers,drop_last=True)
        num_train = len(train_dataset)
        num_val = len(val_dataset)

        if opts.pretrain is None:
            model = MyNet()
        else:
            model = torch.load(opts.pretrain)
        if opts.use_GPU:
            model.to(opts.GPU_id)
        optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
        # optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
        # scheduler=torch.optim.lr_scheduler.StepLR(optimizer,10,0.5)
        best_metric=94.5378
        for e in range(opts.start_epoch, opts.epoch+1):
            t = time.time()
            self.__train(model, train_loader, optimizer, e, num_train, opts)
            t2 = time.time()
            print("Training consumes %.2f second\n" % (t2-t))
            with open(os.path.join(opts.checkpoints_dir, "log.txt"), "a+") as log_file:
                log_file.write("Training consumes %.2f second\n" % (t2-t))
            if e % opts.save_freq==0 or e == opts.epoch+1:
                t = time.time()
                metric = self.__validate(model, val_loader, e, num_val, opts)
                t2 = time.time()
                print("Validation consumes %.2f second\n" % (t2 - t))
                with open(os.path.join(opts.checkpoints_dir, "log.txt"), "a+") as log_file:
                    log_file.write("Validation consumes %.2f second\n" % (t2 - t))
                if best_metric>metric:
                    best_metric = metric
                    print("Epoch %d is now the best epoch with metric %.4f\n"%(e, best_metric))
                    with open(os.path.join(opts.checkpoints_dir, "log.txt"), "a+") as log_file:
                        log_file.write("Epoch %d is now the best epoch with metric %.4f\n"%(e, best_metric))
                self.__save_model(model, e, opts)

#%%
if __name__ == '__main__':
    # 训练网络代码
    args = Args()
    args.set_train_args()  # 获取命令行参数
    train_interface = TrainInterface(args.get_opts())
    train_interface.main()  # 调用训练接口


