import numpy as np
import math

# 距离函数，用欧式距离
def dist(array1, array2):
    d = np.sqrt(np.sum(np.square(array1 - array2)))
    return d


def features(features_list):
    # SCA 同类数据聚合度
    sca_list = []
    center_list = []
    for i in range(len(features_list)):
        center = np.mean(features_list[i], axis=0)  #沿列方向求平均
        center_list.append(center)
        sum = 0
        for j in range(len(features_list[i])):
            sum += dist(features_list[i][j], center)
        sum /= len(features_list[i])
        sca_list.append(sum)
    # 求SCA,不能归一化
    SCA = np.mean(sca_list)

    # SCD 不同类分离程度
    scd_center = np.mean(center_list, axis=0)

    sum_scd = 0
    for i in range(len(center_list)):
        sum_scd += dist(scd_center, center_list[i])
    SCD = sum_scd / len(center_list)

    #RDI 总体指标
    RDI = (SCD - SCA)/max(SCD, SCA)
    return RDI

 
    
