#!/usr/bin/python2.7
#coding:utf-8

# Copyright 2015 Netease Inc. All Rights Reserved.
# Author: tengkz@163.com

"""Fraud detection module for mobile games.

Input ods game records, output ratio and detailed role_id list of fraud users.

Input File format:
    role_id: unique for all servers.
    dt: ods date identification.
    sow_features: sowNum numerical features.
    reap_features: reapNum numerical features.
    Seperated by TAB.

Output:
    Retention info: fraud retention and true retention.
    proj_name.out: file contains role list of fraud users.
"""

import sys
import datetime
import math
import getopt

THETA = 1e-10
THRESHOLD = 0.7

class RoleInfo(object):
    """Integrate role infomation holder.

    Attributes:
        roleID: Role indentification, unique from all servers!
        isFraud: Flag of whether this role is fraud.
        isUsed: Flag of whether this role is used(login days > 1)
        info: Numerical feature value of this role.
        createDt: Create dt of this role.
        loginDtList: List of login dts.
    """
    def __init__(self,role_id,dt):
        self.roleID = role_id
        self.isFraud = False
        self.isUsed = True
        self.info = [1]  #when create,login days = 1
        self.createDt = dt
        self.loginDtList = [dt]  #create dt is first login dt


def check_role_info(roleInfoList):
    """Check whether dt is legal for roleInfoList.

    If there exists login dt earlier than create dt, return False.
    Else, return True

    Args:
        roleInfoList: List contains instances of class RoleInfo.
    
    Returns:
        Whether dt is legal for roleInfoList, True for legal.
    """
    for item in roleInfoList:
        cDt = item.createDt
        for dt in item.loginDtList:
            if dt < cDt:
                return False
    return True


def date_delta(dt1,dt2):
    """Date delta between dt1 and dt2.

    Number of days between dt1(include) and dt2(exclude).

    Args:
        dt1: Begin date.
        dt2: End date.
    Returns:
        Number of days between dt1 and dt2.
    """
    return (datetime.datetime.strptime(dt2,"%Y%m%d") - 
            datetime.datetime.strptime(dt1,"%Y%m%d")).days

def update_date_dict(dateDict,dt,minDt):
    if dt not in dateDict.keys():
        dateDict[dt] = date_delta(minDt,dt)


def construct_login_info(roleInfoList,minDt,maxDt):
    """Construct login info from roleInfoList.

    loginInfo is a matrix, with each row indicate whether a role login in certain
    days, column number is maxDt-minDt+1. '2' for create dt, '1' for other login
    dt, and '0' for no login dt.
    For effectiveness, we use dateDict to preserve date info. Key is date str,
    Value is datedelta from minDt.

    Args:
        roleInfoList: List contains instances of class RoleInfo.
        minDt: min dt of all create days.
        maxDt: max dt of all login days.
    """
    dateWindow = date_delta(minDt,maxDt)
    M = len(roleInfoList)
    dateDict = {}  # for effectiveness
    loginInfo = [[0 for i in xrange(dateWindow+1)] 
            for j in xrange(M)]
    for i in xrange(M):
        item = roleInfoList[i]
        cDt = item.createDt
        update_date_dict(dateDict,cDt,minDt)
        loginInfo[i][dateDict[cDt]] += 1
        for dt in item.loginDtList:
            update_date_dict(dateDict,dt,minDt)
            loginInfo[i][dateDict[dt]] += 1
    return loginInfo


def split_sow_reap(roleInfoList,sowNum,reapNum):
    """Get sow and reap matrix from roleInfoList.

    Extract sow and reap infomation from the roleInfoList, without unused users.

    Args:
        roleInfoList: List contains instances of class RoleInfo.
        sowNum: Number of sow args.
        reapNum: Number of reap args.
    Returns:
        sow: Matrix of sow feature matrix, one row one role.
        reap: Matrix of reap feature matrix, one row one role.
    """
    sow = []
    reap = []
    for item in roleInfoList:
        if item.isUsed:
            sow.append(item.info[0:(1+sowNum)])  # login days and other sow arg
            reap.append(item.info[(1+sowNum):])  # core days and other reap arg
    return sow,reap


def norm(src):
    """Normalize the src matrix.

    Substract mean and divided by standard variance.

    Args:
        src: input matrix for process.
    Returns:
        meanSrc: mean vector.
        sdSrc: standard variance vector.
    """
    M = len(src)  # number of rows
    N = len(src[0])  # number of columns
    meanSrc = []
    sdSrc = []
    for i in xrange(N):
        t = 0.0
        for j in xrange(M):
            t += src[j][i]
        meanSrc.append(t/M)
    for i in xrange(N):
        t = 0.0
        for j in xrange(M):
            t += pow((src[j][i] - meanSrc[i]),2.0)
        sdSrc.append(math.sqrt(t/(M-1)))  # for unbiased estimation
    for i in xrange(N):
        for j in xrange(M):
            src[j][i] = (src[j][i] - meanSrc[i]) / sdSrc[i]
    return meanSrc,sdSrc


def cov_matrix(src):
    """Covariance matrix for normalized matrix.

    Args:
        src: Normalized matrix with mean substracted and std divided.
    Returns:
        cov: Covariance matrix of src.
    """
    M = len(src)
    N = len(src[0])
    cov = [[0 for i in range(N)] for j in range(N)]
    for i in xrange(N):
        for j in xrange(N):
            t = 0.0
            for p in xrange(M):
                t += src[p][i]*src[p][j]
            cov[i][j] = t/(M-1)  # for unbiased estimation
    return cov


def max_engine_vector(src):
    """Get eigen vector of the max eigen value.

    This function is used to help perform PCA which reduce
    dimension from n to 1. Power iteration is used to get 
    eigen value, refs:
    https://en.wikipedia.org/wiki/Power_iteration

    Args:
        src: Covariance matrix.
    Returns:
        Eigen vector of the max eigen value.
    """
    M = len(src)
    x = [1 for i in xrange(M)]
    b = [0 for i in xrange(M)]
    diff = 1e10
    preMaxB = 0.0
    global THETA
    while abs(diff)>THETA:
        for i in xrange(M):
            t = 0.0
            for j in xrange(M):
                t += src[i][j]*x[j]
            b[i] = t
        maxB = abs(b[0])
        for i in xrange(1,M):
            if abs(b[i]) > maxB:
                maxB = abs(b[i])
        for i in xrange(M):
            x[i] = b[i]/maxB
        diff = maxB - preMaxB
        preMaxB = maxB
    # normalize eigen vector.
    sumX = 0.0
    for item in x:
        sumX += item*item
    sumX = math.sqrt(sumX)
    return [item/sumX for item in x]


def pca_one(src,cov):
    """Perform PCA to reduce dimension of src from n to 1.

    Args:
        src: Normalized matrix with mean substracted and std divided.
        cov: Covariance matrix of src.
    Returns:
        pca: One dimension representation of src.
    """
    eng = max_engine_vector(cov)
    rowNum = len(src)
    colNum = len(cov)
    pca = [0 for i in xrange(rowNum)]
    for i in xrange(rowNum):
        for j in xrange(colNum):
            pca[i] += src[i][j]*eng[j]
    return pca


def fraud_detection(sow,reap):
    """Detection fraud role with sow-reap ratio values.

    Compute the sow-reap ratio values, if bigger than THRESHOLD,
    tag as fraud users.

    Args:
        sow: Matrix of sow feature.
        reap: Matrix of reap feature.
    Returns:
        fraudList: List of ids of fraud users. This id is the sequence
        number of all used users whose login days > 1.
    """
    # normalization
    meanSow,sdSow = norm(sow)
    meanReap,sdReap = norm(reap)
    # pca
    covSow = cov_matrix(sow)
    covReap = cov_matrix(reap)
    pcaSow = pca_one(sow,covSow)
    pcaReap = pca_one(reap,covReap)
    # normalization again
    minSow = min(pcaSow)
    minReap = min(pcaReap)
    pcaSow = [item-minSow for item in pcaSow]
    pcaReap = [item-minReap for item in pcaReap]
    # fraud list
    fraudList = []
    for i in xrange(len(sow)):
        if pcaSow[i] != 0 and pcaReap[i]/pcaSow[i]<THRESHOLD:
            fraudList.append(i+1)
    return fraudList


def update_role_info_list(roleInfoList,fraudList):
    """Tag fraud users with roleInfoList and fraudList.

    Two pointers method. One points to the pos in roleInfoList,
    one points to pos in fraudList.

    Args:
        roleInfoList: List contains instances of class RoleInfo.
        fraudList: List of ids of all fraud users.
    Returns:
        None
    """
    t = 0
    i = 0
    for item in fraudList:
        while True:
            if roleInfoList[i].isUsed:
                t += 1
                if t == item:
                    roleInfoList[i].isFraud = True
                    break
            i += 1
        i += 1

def file_process(fileName,sowNum,reapNum):
    """Get info from file.

    Process file record by record, extract basic information.

    Args:
        fileName: Name of file to be processed. File format is like
            role_id, dt, sow_feature1, sow_feature2, ..., reapFeature1, ...
            seperated by '\t'.
        sowNum: Number of sow feature.
        reapNum: Number of reap feature.
    Returns:
        totalCount: Number of different role_ids in file.
        minDt: min dt of all dts.
        maxDt: max dt of all dts.
        roleInfoList: List contains instances of class RoleInfo.
    """
    # Pre definition
    totalCount = 0
    minDt = '20991231'
    maxDt = '00000000'
    roleInfoList = []

    # file process
    preRole = ''
    tmpRoleInfo = RoleInfo("-1","")
    try:
        f = open(fileName,'r')
    except IOError:
        print "ERROR: file %s NOT found! Please check your args." % fileName
        sys.exit(2)
    for line in f:
        lineWhole = line.strip().split("\t")
        if len(lineWhole) != (sowNum+reapNum+2):
            print "ERROR: Column number NOT match!"
            print line
            sys.exit(1)
        role_id = lineWhole[0]
        dt = lineWhole[1]
        sowArg = [float(item) for item in lineWhole[2:(2+sowNum)]]
        reapArg = [float(item) for item in lineWhole[(2+sowNum):]]
        # line process
        if role_id != preRole:  # different role_id
            roleInfoList.append(tmpRoleInfo)
            tmpRoleInfo = RoleInfo(role_id,dt)
            tmpRoleInfo.info.extend(sowArg)
            if sum(reapArg) > 0:
                tmpRoleInfo.info.append(1)  # core days = 1
            else:
                tmpRoleInfo.info.append(0)  # core days = 0
            tmpRoleInfo.info.extend(reapArg)
            preRole = role_id
            totalCount += 1
        else:  # same role_id
            tmpRoleInfo.loginDtList.append(dt)
            tmpRoleInfo.info[0] += 1
            for i in xrange(1,(1+sowNum)):
                tmpRoleInfo.info[i] += sowArg[i-1]  # adds up sow feature
            if sum(reapArg) > 0:
                tmpRoleInfo.info[1+sowNum] += 1
            for i in xrange((2+sowNum),(2+sowNum+reapNum)):  # adds up reap feature
                tmpRoleInfo.info[i] += reapArg[i-2-sowNum]
        # minDt and maxDt
        if dt > maxDt:
            maxDt = dt
        elif dt < minDt:
            minDt = dt
    # add record of first role_id to index 0, which was null before.
    roleInfoList[0] = tmpRoleInfo
    f.close()

    return totalCount, minDt, maxDt, roleInfoList


def tag_used_role(roleInfoList, loginInfo):
    """Tag used role in roleInfoList.

    Tag those login more than 1 days.

    Args:
        roleInfoList: List contains instances of class RoleInfo.
    Returns:
        useCount: Number of role_ids whose login days > 1.
            
    """
    useCount = 0
    for i in xrange(len(roleInfoList)):
        if sum(loginInfo[i]) <= 2:  # after create, no login
            roleInfoList[i].isUsed = False
        else:
            useCount += 1
    return useCount


def info_extract(fileName,sowNum,reapNum):
    """Extract info from file and prepare for fraud detection.

    This contains file_process, construct_login_info and tag used role.

    Returns:
        totalCount: Number of different role_ids in file.
        useCount: Number of used different role_ids in file(login days > 1).
        minDt: min dt of all dts.
        maxDt: max dt of all dts.
    """

    # process file
    totalCount,minDt,maxDt,roleInfoList = file_process(fileName,sowNum,reapNum)

    # check whether date error exists
    if not check_role_info(roleInfoList):
        print "ERROR: Login date earlier than create date!"
        print "ATTENTION: Records should be ordered by role_id and dt successively."
        exit(1)

    # construct loginInfo
    loginInfo = construct_login_info(roleInfoList,minDt,maxDt)

    # detect login days > 1
    useCount = tag_used_role(roleInfoList, loginInfo)

    return totalCount,useCount,minDt,maxDt,roleInfoList,loginInfo


def run(fileName,sowNum=1,reapNum=2):
    """main flow of fraud detection algorithm.

    From raw input file to detailed output info.

    Args:
        fileName: file name to be processed.
        sowNum: number of sow feature.
        reapNum: number of reap feature.
    """
    # extract info from file
    (totalCount,useCount,minDt,maxDt,
            roleInfoList,loginInfo) = info_extract(fileName,sowNum,reapNum)

    # construct sow and reap
    sow,reap = split_sow_reap(roleInfoList,sowNum,reapNum)
    
    # detect fraud
    fraudList = fraud_detection(sow,reap)
    
    
    # update roleInfoList
    update_role_info_list(roleInfoList,fraudList)

    # retention matrix
    dateWindow = date_delta(minDt,maxDt)
    lookValue = [ 
            [0 for i in xrange(dateWindow+1)] 
            for j in xrange(dateWindow+1) 
            ]
    realValue = [ 
            [0 for i in xrange(dateWindow+1)] 
            for j in xrange(dateWindow+1) 
            ]
    newBornLook = [0 for i in xrange(dateWindow+1)]
    newBornReal = [0 for i in xrange(dateWindow+1)]
    
    for i in xrange(len(roleInfoList)):
        roleinfo = roleInfoList[i]
        loginfo = loginInfo[i]
        isFraud = roleinfo.isFraud
        nbDate = loginfo.index(2)
        newBornLook[nbDate] += 1
        if not isFraud:
            newBornReal[nbDate] += 1
        for j in xrange(len(loginfo)):
            if loginfo[j] == 1:
                lookValue[nbDate][j] += 1
                if not isFraud:
                    realValue[nbDate][j] += 1
    
    # conclusions print
    print "Role number %d, reten role number %d, fraud role number %d." % (
            totalCount, useCount, len(fraudList))
    print "Ratio of fraud roles (fraudNum/retenNum) {0:.2f}%.".format(len(fraudList)*100.0/useCount)
    print "Role retention with fraud included and (excluded):"
    for i in xrange(len(newBornLook)):
        for j in xrange(i+1,len(newBornLook)):
            print "".join([
                "{0:.2f}%".format(lookValue[i][j]*100.0/newBornLook[i]),
                "({0:.2f}%)".format(realValue[i][j]*100.0/newBornReal[i])
                ]),
        print
    
    # fraud id print 
    proj_name = "".join(fileName.split('.')[0:-1])
    f = open("%s.out" % proj_name,"w")
    content = []
    for i in xrange(len(roleInfoList)):
        if roleInfoList[i].isFraud:
            content.append(roleInfoList[i].roleID)
    f.write("\n".join(content))


if __name__ == "__main__":
    helpInfo = [
            "NAME",
            "\tfraud_detection",
            "DESCRIPTION",
            "\tInput ods game records, output ratio and detailed role_id list of fraud users.",
            "SYNOPSIS",
            "\tfraud_detection -f proj_name.in -s sow_num -r reap_num",
            "Args:",
            "\t-f,--filename",
            "\t\tinput file name, e.g. ma21.in",
            "\t-s,--sow",
            "\t\tnumber of sow feature",
            "\t-r,--reap",
            "\t\tnumber of reap feature",
            ]
    versionInfo = "Version 1.0."
    try:
        opts, args = getopt.getopt(sys.argv[1:],'f:s:r:hv',
                ['filename=','sow=','reap=','help','version'])
    except getopt.GetoptError, err:
        print str(err)
        sys.exit(2)
    for o,a in opts:
        if o in ('-f','--file'):
            fileName = a
        elif o in ('-s','--sow'):
            sowNum = int(a)
        elif o in ('-r','--reap'):
            reapNum = int(a)
        elif o in ('-h','--help'):
            print "\n".join(helpInfo)
            sys.exit(2)
        elif o in ('-v','--version'):
            print versionInfo
            sys.exit(2)
    if not locals().has_key('fileName'):
        print 'File name NOT defined! Use -h or --help for help.'
        sys.exit(2)
    if not locals().has_key('sowNum'):
        print 'sowNum NOT defined! Use -h or --help for help.'
        sys.exit(2)
    if not locals().has_key('reapNum'):
        print 'reapNum NOT defined! Use -h or --help for help.'
        sys.exit(2)
    run(fileName,sowNum,reapNum)
