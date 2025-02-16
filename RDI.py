import numpy as np
import math

# Distance function with euclidean distance
def dist(array1, array2):
    d = np.sqrt(np.sum(np.square(array1 - array2)))
    return d


def features(features_list):
    # SCA: Degree of aggregation of samples from the same category
    sca_list = []
    center_list = []
    for i in range(len(features_list)):
        center = np.mean(features_list[i], axis=0)  # Averaging along columns
        center_list.append(center)
        sum = 0
        for j in range(len(features_list[i])):
            sum += dist(features_list[i][j], center)
        sum /= len(features_list[i])
        sca_list.append(sum)
    # SCA, not normalized.
    SCA = np.mean(sca_list)

    # SCD: Degree of separation of different classes
    scd_center = np.mean(center_list, axis=0)

    sum_scd = 0
    for i in range(len(center_list)):
        sum_scd += dist(scd_center, center_list[i])
    SCD = sum_scd / len(center_list)

    #RDI
    RDI = (SCD - SCA)/max(SCD, SCA)
    return RDI

 
    
