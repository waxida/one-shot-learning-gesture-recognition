# coding: utf-8
import scipy.io as sio
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

def distance_encludian(v1,v2):
	dist = np.sqrt(np.sum(np.square(v1 - v2)))
	dist = dist / (
	np.sqrt(np.sum(np.square(v1))) + np.sqrt(np.sum(np.square(v2))))
	return dist
def distance_cosine(v1,v2):
	dist = 1-np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
	return dist

if __name__ == '__main__':

    one_shot_class = 10
    n_samples=100

    # features = sio.loadmat("test_samples.mat")
    # features_oneshot = features["foo"]
    # test_features = features_oneshot.reshape([10, 100, -1])
    #
    # features = sio.loadmat("negative_now.mat")
    # negative_features = features["foo"]
    #
    # features = sio.loadmat("register_50.mat")
    # register_features = features["foo"]

    features = sio.loadmat("total_23.mat")
    features_oneshot = features["foo"]
    test_features=features_oneshot[0:1000]
    negative_features=features_oneshot[1000:1100]
    register_features=features_oneshot[1100:]

    test_features = test_features.reshape([10, 100, -1])

    r = []
    for q in range(10):
        r.append(register_features[5 * q])
    contrast_samples = r

    threshold_p=[]
    threshold_r=[]
    threshold_f1=[]
    flag=[]
    flag_confusion_matrix=[]
    threshold_region=np.arange(0.35,0.55,0.001)
    for t in threshold_region :
        total_class_accu = []
        # count_new is confusion matrix of classification results
        count_new=np.zeros([one_shot_class+1,one_shot_class+1])
        for c in range(one_shot_class):
            for j in range(n_samples):
                distance_ou = []
                for i in range(len(contrast_samples)):
                    distance_ou.append(distance_encludian(test_features[c][j],contrast_samples[i]))
                if min(distance_ou)>=t:
                    # print ("the %d th sample of class %d is new"%((j+1),(c+1)))
                    count_new[c][one_shot_class]=count_new[c][one_shot_class]+1
                else:
                    if distance_ou.index(min(distance_ou)) == c:
                        count_new[c][c] = count_new[c][c] + 1
                    else:
                        ind=distance_ou.index(min(distance_ou))
                        count_new[c][ind] = count_new[c][ind] + 1
                        print("the  %d th sample of class %d belongs to class %d"%((j+1),(c+1),(ind+1)))

        # negative

        count = []
        confusion=[]
        for j in range(n_samples):
            distance_ou = []
            for i in range(len(contrast_samples)):
                distance_ou.append(distance_encludian(negative_features[j], contrast_samples[i]))
            if min(distance_ou)>=t:
                count.append(1)
                # print("the %d th sample of class %d is new" % ((j+1), (c + 1)))
            else:
                ind = distance_ou.index(min(distance_ou))
                count_new[one_shot_class][ind]+=1
                count.append(0)
                print "the %d th non-gesture sample is classfied into %d th classes"%((j+1),(ind+1))
        # print ("total %d error"%(count.count(0)))
        count_new[10][10]=100-count.count(0)
        print(count_new)
        flag_confusion_matrix.append(count_new)

        diagonal=0
        last_column=0
        other=0
        last_row=0
        for lsj in range(11):
            for lxj in range(11):
                if lsj==lxj and lxj!=10:
                    diagonal+=count_new[lsj][lxj]
                if lxj==10 and lsj!=10:
                    last_column+=count_new[lsj][lxj]
                if lsj==10 and lxj!=10:
                    last_row+=count_new[lsj][lxj]
                if lsj!=lxj and lsj!=10 and lxj!=10:
                    other+=count_new[lsj][lxj]
        precision=float(diagonal)/(diagonal+other+last_row)
        recall=float(diagonal)/(1000)
        f1=float(2)/(1/precision+1/recall)
        if precision==1 and recall>=0.8:
            flag.append(t)
        print("precision:%.4f"%precision)
        print("recall:%.4f"%recall)
        print("f1:%.4f"%f1)
        print("threshold:%.3f"%t)
        print("---------------------------next starts--------------------------------")
        threshold_p.append(precision)
        threshold_r.append(recall)
        threshold_f1.append(f1)
    #the best threshold
    print("the best result:")
    print ("the max f1-score is %.4f"%(max(threshold_f1)))
    print "the best threshold is %.3f" % (threshold_region[threshold_f1.index(max(threshold_f1))])
    print "the prcision is %.4f "%(threshold_p[threshold_f1.index(max(threshold_f1))])
    print "the recall is %.4f"%(threshold_r[threshold_f1.index(max(threshold_f1))])
    #when precision =1.0
    threshold_region=list(threshold_region)
    print "-------"
    print "when precision==1.0:"
    print "the threshold region is %.3f - %.3f when precision is 1.0"%(flag[0],flag[-1])
    print "the recall region is %.4f - %.4f when precision is 1.0" % (threshold_r[threshold_region.index(flag[0])], threshold_r[threshold_region.index(flag[-1])])
    print "the f1-score region is %.4f - %.4f when precision is 1.0" % (threshold_f1[threshold_region.index(flag[0])], threshold_f1[threshold_region.index(flag[-1])])
    print "the best confusion matrix:"
    print flag_confusion_matrix[threshold_region.index(flag[-1])]

    #visualization
    y1 = threshold_p
    y2=threshold_r
    y3=threshold_f1
    x = threshold_region

    plt.figure(figsize=(7, 5))
    plt.subplot(311)
    plt.plot(x, y1, lw=1.5, label='precision')
    # plt.plot(x, y1, 'bo')
    plt.grid(True)
    plt.legend(loc=0)
    plt.axis('tight')
    plt.ylabel('precision')
    plt.title('precision/recall/F1-score with threshold change')

    plt.subplot(312)
    plt.plot(x, y2,'r', lw=1.5, label='recall')
    # plt.plot(x, y2, 'bo')
    plt.grid(True)
    plt.legend(loc=0)
    plt.axis('tight')
    plt.ylabel('recall')

    plt.subplot(313)
    plt.plot(x,y3,'g', lw = 1.5, label = 'F1-score')
    # plt.plot(x,y3, 'bo')
    plt.grid(True)
    plt.legend(loc = 0)
    plt.xlabel('threshold')
    plt.ylabel('F1-score')
    plt.axis('tight')

    plt.show()