#coding=utf-8
import tensorflow as tf
import numpy as np
import scipy.io as sio
import logging

import socket
import string
import types
from Queue import Queue

import time
import math
import cv2 as cv
import os
import urllib2
import multiprocessing
import traceback

import c3d_biclstm as net
import tensorlayer as tl
import inputs as data

logging.basicConfig(filename='train.log', filemode="w", level=logging.INFO)
flags = tf.app.flags

memory_batch_size=10
FLAGS = flags.FLAGS

log_dir='./logs'
model_save_dir = './models'
model_name='model.ckpt'

global flag2
flag2 = False
# global window_control
# window_control = False
queue2 = Queue(maxsize=10)
bind_ip = '192.168.1.100'
bind_port = 8000

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
server.bind((bind_ip, bind_port))
server.listen(5)
client, add = server.accept()
#print "[***]监听的主机IP是：%s:%d" % (add[0], add[1])


def NNI(dir):
    All  = os.listdir(dir)
    NumL = [int(I[0:-4]) for I in All]
    n = len(All)
    if n < 32:
        Ext = [int(math.ceil(x*32./n)) for x in NumL]
        for i in range(n):
            os.rename(os.path.join(dir, All[i]), os.path.join(dir,'__' + str(Ext[i]) + '.png'))
        for i in range(n):
            os.rename(os.path.join(dir, '__' + str(Ext[i]) + '.png'), os.path.join(dir, str(Ext[i]) + '.png'))
        Dif = list(set(range(1,33)).difference(set(Ext)))
        Dif.sort()
        Ext.sort()
        for i in Dif:
            item = [x>i for x in Ext].index(True)
            obj  = Ext[item]
            os.system('cp '+os.path.join(dir,str(obj)+'.png')+' '+os.path.join(dir,str(i)+'.png'))
    elif n > 32:
        Ext = [int(math.ceil(x*n/32.)) for x in range(1,33)]
        m = len(Ext)
        Dif = list(set(NumL).difference(set(Ext)))
        Dif.sort()
        Ext.sort()
        for i in Dif:
            os.remove(os.path.join(dir, str(i) + '.png'))
        for i in range(m):
            os.rename(os.path.join(dir, str(Ext[i]) + '.png'),os.path.join(dir, '__'+str(i+1) + '.png'))
        for i in range(m):
            os.rename(os.path.join(dir, '__'+str(i+1) + '.png'),os.path.join(dir, str(i+1) + '.png'))

def data_input(testing_datalist,networks,feature,x,y,sess):
    # testing_datalist = './dataset_splits/one_shot_train_22.txt'
    features = []
    batch_size=1
    seq_len=32
    X_test, y_test = data.load_video_list(testing_datalist)
    X_teidx = np.asarray(np.arange(0, len(y_test)), dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)
    for X_indices, y_label_t in tl.iterate.minibatches(X_teidx,
                                                       y_test,
                                                       batch_size,
                                                       shuffle=False):
        # Read data for each batch
        image_path = []
        image_fcnt = []
        image_olen = []
        is_training = []
        for data_a in range(batch_size):
            X_index_a = X_indices[data_a]
            key_str = '%06d' % X_index_a
            image_path.append(X_test[key_str]['videopath'])
            image_fcnt.append(X_test[key_str]['framecnt'])
            image_olen.append(seq_len)
            is_training.append(False)  # Testing
            image_info = zip(image_path, image_fcnt, image_olen, is_training)
        X_data_t = tl.prepro.threading_data([_ for _ in image_info],
                                            data.prepare_isogr_depth_data)
        feed_dict = {x: X_data_t, y: y_label_t}
        dp_dict = tl.utils.dict_to_one(networks.all_drop)
        feed_dict.update(dp_dict)
        feature_ = sess.run( feature,feed_dict=feed_dict)
        features.extend(feature_)
    return features

def download(url, path):
    if url is None:                # 地址若为None则跳过
        pass
    result = urllib2.urlopen(url)  # 打开链接
    if result.getcode() != 200:    # 如果链接不正常，则跳过这个链接
        pass
    else:
        data = result.read()
        with open(path, 'wb') as f:
            f.write(data)
            f.close()

data_queue = multiprocessing.Queue(maxsize=10)
data_show_queue = multiprocessing.Queue(maxsize=10)
data_rec = multiprocessing.Queue(maxsize=10)


# Receive signal
def data_receive():
    keep = True
    while keep:
        try:
            signal = client.recv(1024)
            data_queue.put(signal)

            if signal == "quit":
                break
        except StopIteration:
            print('End')
            break
        except Exception:
            keep = False
            print('Something went wrong with data_receive')
            print(traceback.print_exc())


def distance_encludian(v1, v2):
    dist = np.sqrt(np.sum(np.square(v1 - v2)))
    dist = dist / (
        np.sqrt(np.sum(np.square(v1))) + np.sqrt(np.sum(np.square(v2))))
    return dist
def pre_func(f1,f2):
    distances=[]
    for i in range(len(f1)):
        distances.append(distance_encludian(f1[i],f2))
    if min(distances)>=0.5:
        print("This test sample does not belong to any class,do you want to register a new class?")
        return f2,1
    else:
        pre_label=distances.index(min(distances))
        print("the test sample belongs to class %d" % pre_label)
        return f2,0

# def rename(path):
# 	lines = open(path, 'r')
#     lines = list(lines)
#
def rename_lxj(path):
    images = []
    for image in os.listdir(path):
        images.append(image)
    images = sorted(images, key=lambda x: len(x))
    for i in range(len(images)):
        if i < 9:
            new = path + '/' + str(0) + str(0)+ str(0)+ str(0)+ str(0)+ images[i]
            yuan = path+'/' +images[i]
            os.rename(yuan, new)
        else:
            new = path + '/' + str(0) + str(0)+ str(0)+ str(0)+ images[i]
            yuan = path + '/' + images[i]
            os.rename(yuan, new)
def predict_task():
    num_files = 1
    flag1=True
    # print 'hhhhhh'

    seq_len = 32
    batch_size = 1
    num_classes = 19

    model_prefix = './'

    x = tf.placeholder(tf.float32, [batch_size, seq_len, 112, 112, 3], name='x')
    y = tf.placeholder(tf.int32, shape=[batch_size, ], name='y')
    sess = tf.InteractiveSession()
    feature, networks = net.c3d_biclstm(x, num_classes, False, False)
    feature = feature.outputs
    sess.run(tf.global_variables_initializer())
    load_params = tl.files.load_npz(name='%s/beihang_dataset_birnn_model_epoch_10.npz' % (model_prefix))
    tl.files.assign_params(sess, load_params, networks)

    print("restore done")

    while True:
        if not data_queue.empty():
            command = data_queue.get()
            # print 'The output queue size: {}'.format(data_queue.qsize())

            if command == "quit":
                break
            else:
                file_dir = '/home/e829/Documents/lxj/1128/data/' + 'train_' + str(num_files) + '/'
                if not os.path.isdir(file_dir):
                    os.mkdir(file_dir)
                save_file = os.path.join(os.getcwd(), 'data/train_' + str(num_files) + '/')
                url_path = 'http://192.168.1.104/train_' + str(num_files) + '/'

                nums = string.atoi(command[8:])
                for i in range(nums):
                    url = url_path + str(i + 1) + '.png'
                    path_name = save_file + str(i + 1) + '.png'
                    # 下载
                    download(url, path_name)

                # Data preprocessing
                current_path = os.getcwd()
                nni_path = os.path.join(current_path, 'data/')
                os.chdir(nni_path)
                list_nni = os.listdir(nni_path)
                list_nni.sort(key=lambda x: int(x[6:]))
                file_nni = list_nni[len(list_nni) - 2]

                img_path = os.path.join(nni_path, file_nni)

                starttime3 = time.clock()
                NNI(img_path)
                rename_lxj(img_path)
                os.chdir(current_path)

                with open("./train1.txt", "w") as f:

                    f.write(img_path+'/'+' '+str(32)+ ' '+str(0)+'\n')
                time_preprocess_done = time.clock()
                print ("the NNI time is %f" % (time_preprocess_done - starttime3))

                #lxj

                if flag1:
                    # print("please perform a new gesture！")
                    print("This may be a gesture,do you want to register a new class?")
                    print ("please input:[y/n]")
                    starttime4 = time.clock()
                    data_show_queue.put((-1, img_path,True))
                    last_label = -1
                    while True:
                        # print "while"
                        if not data_rec.empty():
                            input_data = data_rec.get()
                            if input_data == 'y':
                                new_class_feature = data_input('./train1.txt', networks, feature, x, y, sess)
                                # print("Register completed")
                                flag1 = False
                                print "Congratulations!Register a new class successfully!"
                                break
                            elif input_data == 'n':
                                # new_class_feature = new_class_feature
                                print "Please continue!"
                                break
                            else:
                                print "Invalid operation!"
                                break
                        endtime4=time.clock()
                        # print "time"
                        if data_rec.empty() and (endtime4-starttime4)>10.0:
                            print "Time is up! Please continue!"
                            break
                        # break
                        # else:
                        #     print "ok"

                    # new_class_feature=data_input('./train1.txt',networks,feature,x,y,sess)
                    # print("Register completed")
                    # flag1 = False

                else:
                    # print new_class_feature
                    test_features=data_input('./train1.txt',networks,feature,x,y,sess)
                    feature_extraction_done = time.clock()
                    print ("the preprocess and feature extraction time is %f" % (feature_extraction_done - time_preprocess_done))

                    distances = []
                    for i in range(len(new_class_feature)):
                        distances.append(distance_encludian(new_class_feature[i], test_features))
                    if min(distances) >= 0.423:
                        print("The test gesture does not belong to any class,do you want to register a new class?")
                        print ("please input:[y/n]")
                        # cv.destroyAllWindows()
                        starttime5 = time.clock()
                        # data_show_queue.put((-1, img_path))
                        if not last_label == -1:
                            data_show_queue.put((-1, img_path,True))
                            last_label = -1
                        else:
                            data_show_queue.put((-1, img_path,False))
                            last_label = -1
                        while True:
                            if not data_rec.empty():
                                input_data = data_rec.get()
                                if input_data == 'y':
                                    tmp = np.array(test_features)
                                    new_class_feature.append(tmp)
                                    print "Congratulations!Register a new class successfully!"
                                    break
                                elif input_data == 'n':
                                    new_class_feature = new_class_feature
                                    print "Please continue your test!"
                                    break
                                else:
                                    print "Invalid operation!"
                                    break
                            endtime5 = time.clock()
                            if data_rec.empty() and (endtime5 - starttime5) > 10.0:
                                print "Time is up! Please continue!"
                                break

                    else:
                        judge_distance_done = time.clock()
                        print ("the judge distance time is %f" % (judge_distance_done - feature_extraction_done))

                        pre_label = distances.index(min(distances))
                        print("The test gesture belongs to class %d" % (pre_label+1))
                        if last_label == -1:
                            data_show_queue.put((pre_label, img_path, True))
                        else:
                            data_show_queue.put((pre_label, img_path, False))
                        last_label = pre_label
                        endtime3 = time.clock()
                        print ("the predict time is %f" % (endtime3 - judge_distance_done))
                        print "total predict classes time: ", (endtime3 - starttime3)
                        print "-----------------------------------------------------------"

                num_files += 1

# 显示视频分两个函数以防止显示途中地址数据遭篡改造成错误
def videowithlabel(label, videoaddr):
    for i in range(32):
        if flag2 is True:
            break

        img = cv.imread(videoaddr + "/%06d.png" % (i+1))
        if isinstance(img, types.NoneType):
            break
        # if window_control:
        #     cv.destroyAllWindows()
        #     break
        imgc = img.copy()

        if label!=-1:
            font =cv.FONT_HERSHEY_COMPLEX   #使用默认字体
            cv.rectangle(imgc, (10, 5), (310, 35), (0, 0, 255), 2)
            cv.putText(imgc, 'The gesture belongs to new class '+str(label+1), (15, 25), font, 0.45, (0, 0, 0), 1)
            cv.namedWindow("image")
            cv.imshow("image", imgc)
            cv.waitKey(33)
        else:
            cv.namedWindow("image_r")
            cv.imshow("image_r", imgc)
            cv.waitKey(33)

def videoshow():
    while True:
        if flag2 is True:
            break
        if not data_show_queue.empty():
            label,img_path,window_control = data_show_queue.get()


            while True:
                if window_control:
                    window_control = False
                    cv.destroyAllWindows()
                videowithlabel(label,img_path)
                if not data_show_queue.empty():
                    break
                if flag2 is True:
                    break
                    # break


if __name__ == '__main__':

    receive_process = multiprocessing.Process(target=data_receive)
    receive_process.daemon = True
    receive_process.start()

    predict_process = multiprocessing.Process(target=predict_task)
    predict_process.daemon = True
    predict_process.start()

    show_process = multiprocessing.Process(target=videoshow)
    show_process.daemon = True
    show_process.start()

    print ("Main Thread Waiting\n")
    while True:
        data = raw_input()

        data_rec.put(data)

        client.send(data)
        if data == "quit":
            flag2 = True
            cv.waitKey(100)
            break
    client.close()
    server.close()

    root_dir = '/home/e829/Documents/lxj/1128/data/*'
    os.system('rm -rf ' + root_dir)

    print "Done\n"
