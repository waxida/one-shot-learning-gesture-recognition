# -*- coding:utf-8 -*-
import os
import scipy.io as sio
import numpy as np

def get_list(path):
    ret_train = []
    ret_test=[]
    label_train=[]
    label_test=[]
    for root in os.listdir(path):
        i='01'
        # ret_train=[]
        # ret_test=[]
        label=[]
        # for dirs in os.listdir(os.path.join(path,root)):
        ret_train.append(os.path.join(path,root))
        label_train.append(i)
    print ret_train[0][56]
    ret_train=sorted(ret_train,key=lambda x :int(x[57:]))
    print("length of train files is %d"%len(ret_train))
    # print("length of test files is %d" % len(ret_test))
    file=open('negative_bu.txt','w')
    for i in range(len(ret_train)):
        file.write(ret_train[i])
        file.write("/")
        file.write(" ")
        file.write("32")
        file.write(" ")
        file.write(label_train[i])
        file.write("\n")
    file.close()
    print("finish write into train.txt")
    # file=open('test.txt','w')
    # for i in range(len(ret_test)):
    #     file.write(ret_test[i])
    #     # file.write("/")
    #     # file.write(" ")
    #     # file.write("32")
    #     file.write(" ")
    #     file.write(label_test[i])
    #     file.write("\n")
    # file.close()
    # print("finish write into test.txt")
    return 2
def get_list_2(path):
    ret_train = []
    ret_test=[]
    label_train=[]
    label_test=[]
    for root in os.listdir(path):
        i=root[-2:]
        for sample in os.listdir(os.path.join(path,root)):

            # ret_train=[]
            # ret_test=[]
            label=[]
            # for dirs in os.listdir(os.path.join(path,root)):
            ret_train.append(os.path.join(path,root,sample))
            label_train.append(i)
        print ret_train[0][86:]
    ret_train=sorted(ret_train)
    label_train=sorted(label_train)
    # ret_train=sorted(ret_train,key=lambda x :int(x[86:]))
    print("length of train files is %d"%len(ret_train))
    # print("length of test files is %d" % len(ret_test))
    file=open('total_29_classes.txt','w')
    for i in range(len(ret_train)):
        file.write(ret_train[i])
        file.write("/")
        file.write(" ")
        file.write("32")
        file.write(" ")
        file.write(label_train[i])
        file.write("\n")
    file.close()
    print("finish write into train.txt")
    # file=open('test.txt','w')
    # for i in range(len(ret_test)):
    #     file.write(ret_test[i])
    #     # file.write("/")
    #     # file.write(" ")
    #     # file.write("32")
    #     file.write(" ")
    #     file.write(label_test[i])
    #     file.write("\n")
    # file.close()
    # print("finish write into test.txt")
    return 2
def rename_my(path):

    for classes in os.listdir(os.path.join(path)):
        samples=[]
        label='01'
        for sample in os.listdir(os.path.join(path,classes)):
            samples.append(sample)
        samples = sorted(samples, key=lambda x: int(x[:-4]))
        for i in range(len(samples)):
            if i<9:
                yuan = os.path.join(path,classes) + '/' + samples[i]
                new = os.path.join(path,classes)+'/' + str(0)+ str(0)+ str(0)+ str(0)+ str(0)+samples[i]
                os.rename(yuan, new)
            else:
                yuan = os.path.join(path, classes) + '/' + samples[i]
                new = os.path.join(path, classes) + '/' + str(0)+ str(0)+ str(0)+ str(0)+samples[i]
                os.rename(yuan, new)

    return 0
def get_videos_labels_lines(path):
    # Open the file according to the filename
    lines = open(path, 'r')
    lines = list(lines)
    return lines
def getListFiles(path, index):
    ret_train = []
    ret_test = []
    label_train = []
    label_test = []
    for root in os.listdir(path):
        i=root[5:]
        ret=[]
        label=[]
        for dirs in os.listdir(os.path.join(path,root)):
            ret.append(os.path.join(path,root,dirs))
            label.append(i)
        ret=sorted(ret)
        ret_test.extend(ret[15*index:(15*index+15)])
        label_test.extend(label[15*index:(15*index+15)])
        ret_train.extend(ret[:15*index])
        ret_train.extend(ret[(15*index+15):])
        label_train.extend(label[:15*index])
        label_train.extend(label[(15*index+15):])

    print("length of train files is %d"%len(ret_train))
    print("length of test files is %d" % len(ret_test))
    file=open('train.txt','w')
    for i in range(len(ret_train)):
        file.write(ret_train[i])
        file.write("/")
        file.write(" ")
        file.write("32")
        file.write(" ")
        file.write(label_train[i])
        file.write("\n")
    file.close()
    print("finish write into train.txt")
    file=open('test.txt','w')
    for i in range(len(ret_test)):
        file.write(ret_test[i])
        file.write("/")
        file.write(" ")
        file.write("32")
        file.write(" ")
        file.write(label_test[i])
        file.write("\n")
    file.close()
    print("finish write into test.txt")
    return 0
def get_list_from_list(list,index):
    lines=get_videos_labels_lines(list)
    ret_train = []
    ret_test = []
    label_train = []
    label_test = []
    lines=np.array(lines)
    lines=lines.reshape([19,105])
    for j in range(19):
        # i = root[5:]
        ret = []
        label = []
        # for dirs in os.listdir(os.path.join(path, root)):
        #     ret.append(os.path.join(path, root, dirs))
        #     label.append(i)
        # ret = sorted(ret)
        ret_test.extend(lines[j][15 * index:(15 * index + 15)])
        # label_test.extend(label[15 * index:(15 * index + 15)])
        ret_train.extend(lines[j][:15 * index])
        ret_train.extend(lines[j][(15 * index + 15):])
        # label_train.extend(label[:15 * index])
        # label_train.extend(label[(15 * index + 15):])

    print("length of train files is %d" % len(ret_train))
    print("length of test files is %d" % len(ret_test))
    file = open('train.txt', 'w')
    for i in range(len(ret_train)):
        file.write(ret_train[i])
        # file.write("/")
        # file.write(" ")
        # file.write("32")
        # file.write(" ")
        # file.write(label_train[i])
        # file.write("\n")
    file.close()
    print("finish write into train.txt")
    file = open('test.txt', 'w')
    for i in range(len(ret_test)):
        file.write(ret_test[i])
        # file.write("/")
        # file.write(" ")
        # file.write("32")
        # file.write(" ")
        # file.write(label_test[i])
        # file.write("\n")
    file.close()
    print("finish write into test.txt")
    return 0

if __name__ == '__main__':
    path = "C:/lxjcode/train_19c.txt"
    print get_list_from_list(path,0)



