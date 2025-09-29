import pandas as pd
from itertools import combinations
import numpy as np
from scipy.stats import norm
from scipy import stats
import logging
import math
from scipy.stats import chi2
import networkx as nx
from copy import copy
from copy import deepcopy

from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

#_logger = logging.getLogger(__name__)


def HITON_MB(data, target, alaph, is_discrete=True):

    PC, sepset, ci_number = HITON_PC(data, target, alaph, is_discrete)
    # print("PC is:" + str(PC))
    currentMB = PC.copy()
    for x in PC:
        # print("x is: " + str(x))
        PCofPC, _, ci_num2 = HITON_PC(data, x, alaph, is_discrete)
        ci_number += ci_num2
        # print("PCofPC is " + str(PCofPC))
        for y in PCofPC:
            # print("y is " + str(y))
            if y != target and y not in PC:
                conditions_Set = [i for i in sepset[y]]
                conditions_Set.append(x)
                conditions_Set = list(set(conditions_Set))
                ci_number += 1
                pval, dep = cond_indep_test(
                    data, target, y, conditions_Set, is_discrete)
                if pval <= alaph:
                    # print("append is: " + str(y))
                    currentMB.append(y)
                    break

    return [data.columns[i] if isinstance(i, int) else i for i in set(currentMB)], ci_number


def HITON_PC(data, target, alaph, is_discrete=True):
    number, kVar = np.shape(data)
    sepset = [[] for i in range(kVar)]
    variDepSet = []
    candidate_PC = []
    PC = []
    ci_number = 0
    noAdmissionSet = []
    max_k = 3

    # use a list to store variables which are not condition independence with
    # target,and sorted by dep max to min
    candidate_Vars = [i for i in range(kVar) if i != target]
    for x in candidate_Vars:
        ci_number += 1
        pval_gp, dep_gp = cond_indep_test(
            data, target, x, [], is_discrete)
        if pval_gp <= alaph:
            variDepSet.append([x, dep_gp])

    # sorted by dep from max to min
    variDepSet = sorted(variDepSet, key=lambda x: x[1], reverse=True)
    # print(variDepSet)

    # get number by dep from max to min
    for i in range(len(variDepSet)):
        candidate_PC.append(variDepSet[i][0])
    # print(candidate_PC)

    """ sp """
    for x in candidate_PC:

        PC.append(x)
        PC_index = len(PC)
        # if new x add will be removed ,test will not be continue,so break the
        # following circulate to save time ,but i don't not why other index
        # improve
        breakFlagTwo = False

        while PC_index >= 0:
            #  reverse traversal PC,and use PC_index as a pointer of PC
            PC_index -= 1
            y = PC[PC_index]
            breakFlag = False
            conditions_Set = [i for i in PC if i != y]

            if len(conditions_Set) >= max_k:
                Slength = max_k
            else:
                Slength = len(conditions_Set)

            for j in range(Slength + 1):
                SS = subsets(conditions_Set, j)
                for s in SS:
                    ci_number += 1
                    conditions_test_set = [i for i in s]
                    pval_rm, dep_rm = cond_indep_test(
                        data, target, y, conditions_test_set, is_discrete)
                    if pval_rm > alaph:
                        sepset[y] = [i for i in conditions_test_set]
                        # if new x add will be removed ,test will not be
                        # continue
                        if y == x:
                            breakFlagTwo = True
                        PC.remove(y)
                        breakFlag = True
                        break

                if breakFlag:
                    break
            if breakFlagTwo:
                break

    return list(set(PC)), sepset, ci_number

def IAMB(data, target, alaph, is_discrete=True):
    number, kVar = np.shape(data)
    CMB = []
    ci_number = 0
    # forward circulate phase
    circulate_Flag = True
    while circulate_Flag:
        # if not change, forward phase of IAMB is finished.
        circulate_Flag = False
        # tem_dep pre-set infinite negative.
        temp_dep = -(float)("inf")
        y = None
        variables = [i for i in range(kVar) if i != target and i not in CMB]

        for x in variables:
            ci_number += 1
            pval, dep = cond_indep_test(data, target, x, CMB, is_discrete)
            # print("target is:",target,",x is: ", x," CMB is: ", CMB," ,pval is: ",pval," ,dep is: ", dep)

            # chose maxsize of f(X:T|CMB)
            if pval <= alaph:
                if dep > temp_dep:
                    temp_dep=dep
                    y=x

        # if not condition independence the node,appended to CMB
        if y is not None:
            # print('appended is :'+str(y))
            CMB.append(y)
            circulate_Flag = True

    # backward circulate phase
    CMB_temp = CMB.copy()
    for x in CMB_temp:
        # exclude variable which need test p-value
        condition_Variables=[i for i in CMB if i != x]
        ci_number += 1
        pval, dep = cond_indep_test(data,target, x, condition_Variables, is_discrete)
        # print("target is:", target, ",x is: ", x, " condition_Variables is: ", condition_Variables, " ,pval is: ", pval, " ,dep is: ", dep)
        if pval > alaph:
            # print("removed variables is: " + str(x))
            CMB.remove(x)

    return [data.columns[i] if isinstance(i, int) else i for i in set(CMB)], ci_number


def LRH(data, target, alaph, is_discrete=True):
    ci_number = 0
    number, kVar = np.shape(data)
    max_k = 3
    M = []
    while True:
        # selection
        M1 = []
        x_dep_set = []
        variables = [i for i in range(kVar) if i != target and i not in M]
        for x in variables:
            ci_number += 1
            pval, dep = cond_indep_test(data, target, x, M, is_discrete)
            if pval <= alaph:
                M1.append(x)
                x_dep_set.append([x,dep])

        # exclusion
        if M1 == []:
            break
        elif len(M1) == 1:
            M.append(M1[0])
            continue
        M2 = []
        # print("M is: " + str(M))
        # print("M1 is: " + str(M1))
        for x in M1:
            # print("x is: " + str(x))
            NX = []
            vari_set = [i for i in M1 if i != x]
            for y in vari_set:
                ci_number += 1
                pval, _ = cond_indep_test(data, x, y, M, is_discrete)
                if pval <= alaph:
                    NX.append(y)
            # print("NX is:" + str(NX))
            Nlength = len(NX)
            if Nlength > max_k:
                Nlength = 3
            break_flag = False
            for j in range(Nlength+1):
                Z_set = subsets(NX, j)
                for Z in Z_set:
                    conditionset = list(set(Z).union(set(M)))
                    ci_number += 1
                    pval, _ = cond_indep_test(data, target, x, conditionset, is_discrete)
                    # print("pval is: " + str(pval) + " ,x is: " + str(x) + " ,conditionset is: " + str(conditionset))
                    if pval > alaph:
                        break_flag = True
                        break
                if break_flag:
                    break
            if not break_flag:
                M2.append(x)
                # print("M2 append is: " + str(M2))
        # print("M2 is: " + str(M2))
        Y = []

        if M2 == []:
            x_dep_set = sorted(x_dep_set, key=lambda x: x[1], reverse=True)
            # print("-x_dep_set is: " + str(x_dep_set))
            if x_dep_set != []:
                dep_max = x_dep_set[0][1]
                for m in x_dep_set:
                    if m[1] == dep_max:
                        Y.append(m[0])
                    else:
                        break
        else:
            x_dep_set = []
            for x in M2:
                ci_number += 1
                pval, dep = cond_indep_test(data, target, x, M, is_discrete)
                if pval <= alaph:
                    x_dep_set.append([x, dep])
            x_dep_set = sorted(x_dep_set, key=lambda x: x[1], reverse=True)
            # print("--x_dep_set is: " + str(x_dep_set))
            if x_dep_set != []:
                dep_max = x_dep_set[0][1]
                for m in x_dep_set:
                    if m[1] == dep_max:
                        Y.append(m[0])
                    else:
                        break

        # M3 = [i for i in M1 if i not in M2]
        M = list(set(M).union(set(Y)))

    # print("-M is: " + str(M))
    M_temp = M.copy()
    for x in M_temp:
        conditionset = [i for i in M if i != x]
        ci_number += 1
        pval, _ = cond_indep_test(data, target, x, conditionset, is_discrete)
        # print("pval is: " + str(pval) + " , x is: " + str(x))
        if pval > alaph:
            M.remove(x)

    return [data.columns[i] if isinstance(i, int) else i for i in set(M)], ci_number

def BAMB(data, target, alaph, is_discrete=True):
    ci_number = 0
    number, kVar = np.shape(data)
    max_k = 3
    CPC = []
    TMP = [i for i in range(kVar) if i != target]
    sepset = [[] for i in range(kVar)]
    CSPT = [[] for i in range(kVar)]
    variDepSet = []
    SP = [[] for i in range(kVar)]
    PC = []

    for x in TMP:
        ci_number += 1
        pval_f, dep_f = cond_indep_test(data, target, x, [], is_discrete)
        if pval_f > alaph:
            sepset[x] = []
        else:
            variDepSet.append([x, dep_f])

    variDepSet = sorted(variDepSet, key=lambda x: x[1], reverse=True)
    """step one: Find the candidate set of PC and candidate set of spouse"""

    # print("variDepSet" + str(variDepSet))
    for variIndex in variDepSet:
        A = variIndex[0]
        # print("A is: " + str(A))
        Slength = len(CPC)
        if Slength > max_k:
            Slength = 3
        breakFlag = False
        for j in range(Slength + 1):
            ZSubsets = subsets(CPC, j)
            for Z in ZSubsets:
                ci_number += 1
                convari = [i for i in Z]
                pval_TAZ, dep_TAZ = cond_indep_test(
                    data, target, A, convari, is_discrete)
                if pval_TAZ > alaph:
                    sepset[A] = convari
                    breakFlag = True
                    # print("ZZZ")
                    break
            if breakFlag:
                break

        if not breakFlag:
            CPC_ReA = CPC.copy()
            B_index = len(CPC_ReA)
            CPC.append(A)
            breakF = False
            while B_index > 0:
                B_index -= 1
                B = CPC_ReA[B_index]
                flag1 = False

                conditionSet = [i for i in CPC_ReA if i != B]
                Clength = len(conditionSet)
                if Clength > max_k:
                    Clength = max_k
                for j in range(Clength + 1):
                    CSubsets = subsets(conditionSet, j)
                    for Z in CSubsets:
                        ci_number += 1
                        convari = [i for i in Z]
                        pval_TBZ, dep_TBZ = cond_indep_test(
                            data, target, B, convari, is_discrete)
                        # print("pval_TBZ: " + str(pval_TBZ))
                        if pval_TBZ >= alaph:

                            CPC.remove(B)
                            CSPT[B] = []
                            sepset[B] = convari

                            flag1 = True
                            if B == A:
                                breakF = True
                    if flag1:
                        break
                if breakF:
                    break

            CSPT[A] = []
            pval_CSPT = []

            # add candidate of spouse

            # print("sepset: " + str(sepset))
            for C in range(kVar):
                if C == target or C in CPC:
                    continue
                conditionSet = [i for i in sepset[C]]
                conditionSet.append(A)
                conditionSet = list(set(conditionSet))

                ci_number += 1
                pval_CAT, _ = cond_indep_test(
                    data, target, C, conditionSet, is_discrete)
                if pval_CAT <= alaph:
                    CSPT[A].append(C)
                    pval_CSPT.append([C, pval_CAT])

            """step 2-1"""

            pval_CSPT = sorted(pval_CSPT, key=lambda x: x[1], reverse=False)
            SP[A] = []
            # print("CSPT-: " +str(CSPT))
            # print("pval_CSPT is: " + str(pval_CSPT))

            for pCSPT_index in pval_CSPT:
                E = pCSPT_index[0]
                # print("E is:" + str(E))

                SP[A].append(E)
                index_spa = len(SP[A])
                breakflag_spa = False
                # print("SP[A] is: " +str(SP[A]))
                while index_spa >= 0:
                    index_spa -= 1
                    x = SP[A][index_spa]
                    breakFlag = False
                    # print("x is:" + str(x))

                    ZAllconditionSet = [i for i in SP[A] if i != x]
                    # print("ZAllconditionSet is:" + str(ZAllconditionSet))
                    for Z in ZAllconditionSet:
                        conditionvari = [Z]
                        if A not in conditionvari:
                            conditionvari.append(A)
                        ci_number += 1
                        pval_TXZ, _ = cond_indep_test(
                            data, target, x, conditionvari, is_discrete)
                        # print("x is: " + str(x) + "conditionvari: " + str(conditionvari) + " ,pval_TXZ is: " + str(pval_TXZ))
                        if pval_TXZ > alaph:
                            # print("spa is: " + str(SP[A]) + " .remove x is: " + str(x) + " ,Z is: " + str(conditionvari))
                            SP[A].remove(x)
                            breakFlag = True

                            if x == E:
                                breakflag_spa = True
                            break
                    if breakFlag:
                        break
                if breakflag_spa:
                    break

            """step 2-2"""
            # remove x from pval_CSPT
            pval_CSPT_new = []
            plength = len(pval_CSPT)
            for i in range(plength):
                if pval_CSPT[i][0] in SP[A]:
                    pval_CSPT_new.append(pval_CSPT[i])

            CSPT[A] = SP[A]
            SP[A] = []
            # print("CSPT-: " + str(CSPT))
            # print("2222222pval_CSPT_new is: " + str(pval_CSPT_new))

            for pCSPT_index in pval_CSPT_new:
                E = pCSPT_index[0]
                # print("E2 is:" + str(E))

                SP[A].append(E)
                index_spa = len(SP[A])
                breakflag_spa = False
                # print("SP[A] is: " + str(SP[A]))
                while index_spa >= 0:
                    index_spa -= 1
                    x = SP[A][index_spa]

                    breakFlag = False
                    # print("x is:" + str(x))
                    ZAllSubsets = list(set(CPC).union(set(SP[A])))
                    # print("CPC is: " + str(CPC) + " , SP[A] is: " + str(SP[A]) + " ,A is" + str(A) + " ,x is:" + str(x) + " ,ZA is: " + str(ZAllSubsets))
                    ZAllSubsets.remove(x)
                    ZAllSubsets.remove(A)
                    # print("-ZALLSubsets has: " + str(ZAllSubsets))
                    Zalength = len(ZAllSubsets)
                    if Zalength > max_k:
                        Zalength = max_k
                    for j in range(Zalength + 1):
                        ZaSubsets = subsets(ZAllSubsets, j)
                        for Z in ZaSubsets:
                            Z = [i for i in Z]
                            ci_number += 1
                            pval_TXZ, _ = cond_indep_test(
                                data, A, x, Z, is_discrete)
                            # print("Z is: " + str(Z) + " ,A is: " + str(A) + " ,x is: " + str(x) + " ,pval_txz is: " + str(pval_TXZ))
                            if pval_TXZ > alaph:
                                # print("spa is:" + str(SP[A]) + " .remove x is: " + str(x) + " ,Z is: " + str(Z))
                                SP[A].remove(x)
                                breakFlag = True
                                if x == E:
                                    breakflag_spa = True
                                break
                        if breakFlag:
                            break
                    if breakflag_spa:
                        break

            """ step 2-3"""
            pval_CSPT_fin = []
            plength = len(pval_CSPT)
            for i in range(plength):
                if pval_CSPT[i][0] in SP[A]:
                    pval_CSPT_fin.append(pval_CSPT[i])

            CSPT[A] = SP[A]
            SP[A] = []
            # print("CSPT-: " +str(CSPT))
            # print("2222222pval_CSPT_fin is: " + str(pval_CSPT_fin))

            for pCSPT_index in pval_CSPT_fin:
                E = pCSPT_index[0]
                # print("E3 is:" + str(E))

                SP[A].append(E)
                index_spa = len(SP[A])
                breakflag_spa = False
                # print("SP[A] is: " + str(SP[A]))
                while index_spa >= 0:
                    index_spa -= 1
                    x = SP[A][index_spa]
                    breakFlag = False

                    # print("x is:" + str(x))
                    ZAllSubsets = list(set(CPC).union(set(SP[A])))
                    ZAllSubsets.remove(x)
                    ZAllSubsets.remove(A)
                    Zalength = len(ZAllSubsets)
                    # print("=-ZALLSubsets has: " + str(ZAllSubsets))
                    if Zalength > max_k:
                        Zalength = max_k
                    for j in range(Zalength + 1):
                        ZaSubsets = subsets(ZAllSubsets, j)
                        # print("ZzSubsets is: " + str(ZaSubsets))
                        for Z in ZaSubsets:
                            Z = [i for i in Z]
                            Z.append(A)
                            # print("Z in ZaSubsets is: " + str(Z))
                            ci_number += 1
                            pval_TXZ, _ = cond_indep_test(
                                data, target, x, Z, is_discrete)
                            # print("-Z is: " + str(Z) + " ,x is: " + str(x) + " ,pval_txz is: " + str(
                            #     pval_TXZ))
                            if pval_TXZ >= alaph:
                                # print("spa is:" + str(SP[A]) + " .remove x is: " + str(x) + " ,Z is: " + str(Z))
                                SP[A].remove(x)
                                if x == E:
                                    breakflag_spa = True
                                breakFlag = True
                                break
                        if breakFlag:
                            break
                    if breakflag_spa:
                        break
            # print("SP[A]------: " + str(SP[A]))
            CSPT[A] = SP[A]
            # print("CSPT is: " + str(CSPT))

            """step3: remove false positives from the candidate set of PC"""

            CPC_temp = CPC.copy()
            x_index = len(CPC_temp)
            A_breakFlag = False
            # print("-CPC-: " + str(CPC))
            while x_index >= 0:
                x_index -= 1
                x = CPC_temp[x_index]
                flag2 = False
                ZZALLsubsets = [i for i in CPC if i != x]
                # print("xx is: " + str(x) + ", ZZALLsubsets is: " + str(ZZALLsubsets ))
                Zlength = len(ZZALLsubsets)
                if Zlength > max_k:
                    Zlength = max_k
                for j in range(Zlength + 1):
                    Zzsubsets = subsets(ZZALLsubsets, j)
                    for Z in Zzsubsets:
                        conditionSet = [
                            i for y in Z for i in CSPT[y] if i not in CPC]
                        conditionSet = list(set(conditionSet).union(set(Z)))
                        # print("conditionSet: " + str(conditionSet))
                        ci_number += 1
                        pval, _ = cond_indep_test(
                            data, target, x, conditionSet, is_discrete)
                        if pval >= alaph:
                            # print("remove x is: " + str(x) + " , pval is: " + str(pval) + " ,conditionset is: " + str(conditionSet))
                            CPC.remove(x)
                            CSPT[x] = []
                            flag2 = True
                            if x == A:
                                A_breakFlag = True
                            break
                    if flag2:
                        break
                if A_breakFlag:
                    break

    # print("SP is:" + str(SP))
    spouseT = [j for i in CPC for j in CSPT[i]]
    MB = list(set(CPC).union(set(spouseT)))
    return [data.columns[i] if isinstance(i, int) else i for i in set(MB)], ci_number


def subsets(nbrs, k):
    return set(combinations(nbrs, k))

def cond_indep_test(data, target, var, cond_set=[], is_discrete=True, alpha=0.01):
    if is_discrete:
        pval, dep = g2_test_dis(data, target, var, cond_set,alpha)
        # if selected:
        #     _, pval, _, dep = chi_square_test(data, target, var, cond_set, alpha)
        # else:
        # _, _, dep, pval = chi_square(target, var, cond_set, data, alpha)
    else:
        CI, dep, pval = cond_indep_fisher_z(data, target, var, cond_set, alpha)
    return pval, dep

def g_square_dis(dm, x, y, s, alpha, levels):
    """G square test for discrete data.

    Args:
        dm: the data matrix to be used (as a numpy.ndarray).
        x: the first node (as an integer).
        y: the second node (as an integer).
        s: the set of neibouring nodes of x and y (as a set()).
        levels: levels of each column in the data matrix
            (as a list()).

    Returns:
        p_val: the p-value of conditional independence.
    """

    def _calculate_tlog(x, y, s, dof, levels, dm):
        prod_levels = np.prod(list(map(lambda x: levels[x], s)))
        nijk = np.zeros((levels[x], levels[y], prod_levels))
        s_size = len(s)
        z = []
        for z_index in range(s_size):
            z.append(s.pop())
            pass
        for row_index in range(dm.shape[0]):
            i = dm[row_index, x]
            j = dm[row_index, y]
            k = []
            k_index = 0
            for s_index in range(s_size):
                if s_index == 0:
                    k_index += dm[row_index, z[s_index]]
                else:
                    lprod = np.prod(list(map(lambda x: levels[x], z[:s_index])))
                    k_index += (dm[row_index, z[s_index]] * lprod)
                    pass
                pass
            nijk[i, j, k_index] += 1
            pass
        nik = np.ndarray((levels[x], prod_levels))
        njk = np.ndarray((levels[y], prod_levels))
        for k_index in range(prod_levels):
            nik[:, k_index] = nijk[:, :, k_index].sum(axis = 1)
            njk[:, k_index] = nijk[:, :, k_index].sum(axis = 0)
            pass
        nk = njk.sum(axis = 0)
        tlog = np.zeros((levels[x], levels[y], prod_levels))
        tlog.fill(np.nan)
        for k in range(prod_levels):
            tx = np.array([nik[:, k]]).T
            ty = np.array([njk[:, k]])
            tdijk = tx.dot(ty)
            tlog[:, :, k] = nijk[:, :, k] * nk[k] / tdijk
            pass
        return (nijk, tlog)

    #_logger.debug('Edge %d -- %d with subset: %s' % (x, y, s))
    row_size = dm.shape[0]
    s_size = len(s)
    print("x:", x, type(x))
    print("y:", y, type(y))
    print("s:", s)
    dof = ((levels[x] - 1) * (levels[y] - 1)
           * np.prod([levels[si] for si in s]))

    # row_size_required = 5 * dof
    # if row_size < row_size_required:
    #     _logger.warning('Not enough samples. %s is too small. Need %s.'
    #                     % (str(row_size), str(row_size_required)))
    #     p_val = 1
    #     dep = 0
    #     return p_val, dep

    nijk = None
    if s_size < 5:
        if s_size == 0:
            nijk = np.zeros((levels[x], levels[y]))
            for row_index in range(row_size):
                i = dm[row_index, x]
                j = dm[row_index, y]
                nijk[i, j] += 1
                pass
            tx = np.array([nijk.sum(axis = 1)]).T
            ty = np.array([nijk.sum(axis = 0)])
            tdij = tx.dot(ty)
            tlog = nijk * row_size / tdij
            pass
        if s_size > 0:
            nijk, tlog = _calculate_tlog(x, y, s, dof, levels, dm)
            pass
        pass
    else:
        # s_size >= 5
        nijk = np.zeros((levels[x], levels[y], 1))
        i = dm[0, x]
        j = dm[0, y]
        k = []
        for z in s:
            k.append(dm[:, z])
            pass
        k = np.array(k).T
        parents_count = 1
        parents_val = np.array([k[0, :]])
        nijk[i, j, parents_count - 1] = 1
        for it_sample in range(1, row_size):
            is_new = True
            i = dm[it_sample, x]
            j = dm[it_sample, y]
            tcomp = parents_val[:parents_count, :] == k[it_sample, :]
            for it_parents in range(parents_count):
                if np.all(tcomp[it_parents, :]):
                    nijk[i, j, it_parents] += 1
                    is_new = False
                    break
                pass
            if is_new is True:
                parents_count += 1
                parents_val = np.r_[parents_val, [k[it_sample, :]]]
                nnijk = np.zeros((levels[x], levels[y], parents_count))
                for p in range(parents_count - 1):
                    nnijk[:, :, p] = nijk[:, :, p]
                    pass
                nnijk[i, j, parents_count - 1] = 1
                nijk = nnijk
                pass
            pass
        nik = np.ndarray((levels[x], parents_count))
        njk = np.ndarray((levels[y], parents_count))
        for k_index in range(parents_count):
            nik[:, k_index] = nijk[:, :, k_index].sum(axis = 1)
            njk[:, k_index] = nijk[:, :, k_index].sum(axis = 0)
            pass
        nk = njk.sum(axis = 0)
        tlog = np.zeros((levels[x], levels[y], parents_count))
        tlog.fill(np.nan)
        for k in range(parents_count):
            tx = np.array([nik[:, k]]).T
            ty = np.array([njk[:, k]])
            tdijk = tx.dot(ty)
            tlog[:, :, k] = nijk[:, :, k] * nk[k] / tdijk
            pass
        pass
    log_tlog = np.log(tlog)
    G2 = np.nansum(2 * nijk * log_tlog)
    # _logger.debug('dof = %d' % dof)
    # _logger.debug('nijk = %s' % nijk)
    # _logger.debug('tlog = %s' % tlog)
    # _logger.debug('log(tlog) = %s' % log_tlog)
    #_logger.debug('G2 = %f' % G2)
    if dof == 0:
        # dof can be 0 when levels[x] or levels[y] is 1, which is
        # the case that the values of columns x or y are all 0.
        p_val = 1
        G2 = 0
    else:
        p_val = chi2.sf(G2, dof)
        # print("p-value:", p_val)
    #_logger.info('p_val = %s' % str(p_val))

    if p_val > alpha:
        dep = 0
    else:
        dep = abs(G2)
    return p_val, dep


def g2_test_dis(data_matrix, x, y, s, alpha, **kwargs):
    s1 = sorted([i for i in s])
    levels = []
    data_matrix = np.array(data_matrix, dtype=int)
    # print(data_matrix)
    # print("x: ", x, " ,y: ", y, " ,s: ", s1)
    if 'levels' in kwargs:
        levels = kwargs['levels']
    else:
        levels = np.amax(data_matrix, axis=0) + 1
    return g_square_dis(data_matrix, x, y, s1, alpha, levels)


def get_partial_matrix(S, X, Y):
    S = S[X, :]
    S = S[:, Y]
    return S


def partial_corr_coef(S, i, j, Y):
    S = np.matrix(S)
    X = [i, j]
    inv_syy = np.linalg.inv(get_partial_matrix(S, Y, Y))
    i2 = 0
    j2 = 1
    S2 = get_partial_matrix(S, X, X) - get_partial_matrix(S, X, Y) * inv_syy * get_partial_matrix(S, Y, X)
    c = S2[i2, j2]
    r = c / np.sqrt((S2[i2, i2] * S2[j2, j2]))

    return r


def cond_indep_fisher_z(data, var1, var2, cond=[], alpha=0.05):

    """
    COND_INDEP_FISHER_Z Test if var1 indep var2 given cond using Fisher's Z test
    CI = cond_indep_fisher_z(X, Y, S, C, N, alpha)
    C is the covariance (or correlation) matrix
    N is the sample size
    alpha is the significance level (default: 0.05)
    transfromed from matlab
    See p133 of T. Anderson, "An Intro. to Multivariate Statistical Analysis", 1984

    Parameters
    ----------
    data: pandas Dataframe
        The dataset on which to test the independence condition.

    var1: str
        First variable in the independence condition.

    var2: str
        Second variable in the independence condition

    cond: list
        List of variable names in given variables.

    Returns
    -------

    CI: int
        The  conditional independence of the fisher z test.
    r: float
        partial correlation coefficient
    p_value: float
        The p-value of the test
    """

    N, k_var = np.shape(data)
    list_z = [var1, var2] + list(cond)
    list_new = [data.columns.get_loc(a) if isinstance(a, str) else int(a) for a in list_z]
    data_array = np.array(data)
    array_new = np.transpose(np.matrix(data_array[:, list_new]))
    cov_array = np.cov(array_new)
    size_c = len(list_new)
    X1 = 0
    Y1 = 1
    S1 = [i for i in range(size_c) if i != 0 and i != 1]
    r = partial_corr_coef(cov_array, X1, Y1, S1)
    z = 0.5 * np.log((1+r) / (1-r))
    z0 = 0
    W = np.sqrt(N - len(S1) - 3) * (z - z0)
    cutoff = norm.ppf(1 - 0.5 * alpha)
    if abs(W) < cutoff:
        CI = 1
    else:
        CI = 0
    p = norm.cdf(W)
    r = abs(r)

    return CI, r, p

def chi_square_test(df, var1, var2, condition_vars=None, alaph=0.01, **kwargs):
    """
    Test for the independence condition (var1 _|_ var2 | condition_vars) in df.

    Parameters
    ----------
    df: pandas Dataframe
        The dataset on which to test the independence condition.

    var1: str
        First variable in the independence condition.

    var2: str
        Second variable in the independence condition

    condition_vars: list
        List of variable names in given variables.

    Returns
    -------
    chi_stat: float
        The chi-square statistic for the test.

    p_value: float
        The p-value of the test

    dof: int
        Degrees of Freedom
    """
    var1 = str(var1)
    var2 = str(var2)
    condition_vars = [str(i) for i in condition_vars]
    if "state_names" in kwargs.keys():
        state_names = kwargs["state_names"]
    else:
        state_names = {
            var_name: df.loc[:, var_name].unique() for var_name in df.columns
        }
    num_params = (
        (len(state_names[var1]) - 1)
        * (len(state_names[var2]) - 1)
        * np.prod([len(state_names[z]) for z in condition_vars])
    )
    sufficient_data = len(df) >= num_params * 5
    if not condition_vars:
        observed = pd.crosstab(df[str(var1)], df[str(var2)])
        chi_stat, p_value, dof, expected = stats.chi2_contingency(observed)

    else:
        observed_combinations = df.groupby(condition_vars).size().reset_index()

        chi_stat = 0
        dof = 0
        for combination in range(len(observed_combinations)):
            df_conditioned = df.copy()
            for condition_var in condition_vars:
                df_conditioned = df_conditioned.loc[df_conditioned.loc[:,
                                                                       condition_var] == observed_combinations.loc[combination,
                                                                                                                   condition_var]]
            observed = pd.crosstab(df_conditioned[var1], df_conditioned[var2])
            chi, _, freedom, _ = stats.chi2_contingency(observed)
            chi_stat += chi
            dof += freedom
        p_value = 1.0 - stats.chi2.cdf(x=chi_stat, df=dof)

    if dof <= 0:
        dof = 1
    # print("chi_stat is ", chi_stat, "dof is ", dof)
    if math.isnan(p_value):
        chi_stat = 1.0
        dep = 2.0 + chi_stat / num_params
    elif p_value > alaph:
        dep = -2.0 - chi_stat / num_params
    else:
        dep = 2.0 + chi_stat / num_params

    # n = df.shape[0]
    # rmsea = sqrt((chi_stat / dof - 1) / (n - 1))

    p_value = round(p_value, 6)
    # if p_value > alaph:
    #     p_valueX, p_valueY = str(p_value).split('.')
    #     p_value = float(p_valueX + '.' + p_valueY[0:2])

    return chi_stat, p_value, dof, 1-p_value



def chi_square(X, Y, Z, data, alpha, **kwargs):
    """
    Chi-square conditional independence test.
    Tests the null hypothesis that X is independent from Y given Zs.

    This is done by comparing the observed frequencies with the expected
    frequencies if X,Y were conditionally independent, using a chisquare
    deviance statistic. The expected frequencies given independence are
    `P(X,Y,Zs) = P(X|Zs)*P(Y|Zs)*P(Zs)`. The latter term can be computed
    as `P(X,Zs)*P(Y,Zs)/P(Zs).

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set
    Y: int, string, hashable object
        A variable name contained in the data set, different from X
    Zs: list of variable names
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    Returns
    -------
    chi2: float
        The chi2 test statistic.
    p_value: float
        The p_value, i.e. the probability of observing the computed chi2
        statistic (or an even higher value), given the null hypothesis
        that X _|_ Y | Zs.
    sufficient_data: bool
        A flag that indicates if the sample size is considered sufficient.
        As in [4], require at least 5 samples per parameter (on average).
        That is, the size of the data set must be greater than
        `5 * (c(X) - 1) * (c(Y) - 1) * prod([c(Z) for Z in Zs])`
        (c() denotes the variable cardinality).


    References
    ----------
    [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
    Section 18.2.2.3 (page 789)
    [2] Neapolitan, Learning Bayesian Networks, Section 10.3 (page 600ff)
        http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf
    [3] Chi-square test https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test#Test_of_independence
    [4] Tsamardinos et al., The max-min hill-climbing BN structure learning algorithm, 2005, Section 4

    Examples
    --------
    # >>> import pandas as pd
    # >>> import numpy as np
    # >>> from pgmpy.estimators import ConstraintBasedEstimator
    # >>> data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
    # >>> data['E'] = data['A'] + data['B'] + data['C']
    # >>> c = ConstraintBasedEstimator(data)
    # >>> print(c.test_conditional_independence('A', 'C'))  # independent
    # True
    # >>> print(c.test_conditional_independence('A', 'B', 'D'))  # independent
    # True
    # >>> print(c.test_conditional_independence('A', 'B', ['D', 'E']))  # dependent
    False
    """
    X = str(X)
    Y = str(Y)
    Z = [str(i) for i in Z]
    if isinstance(Z, (frozenset, list, set, tuple)):
        Z = list(Z)
    else:
        Z = [Z]

    if "state_names" in kwargs.keys():
        state_names = kwargs["state_names"]
    else:
        state_names = {
            var_name: data.loc[:, var_name].unique() for var_name in data.columns
        }

    num_params = (
        (len(state_names[X]) - 1)
        * (len(state_names[Y]) - 1)
        * np.prod([len(state_names[z]) for z in Z])
    )
    sufficient_data = len(data) >= num_params * 5

    if not sufficient_data:
        # warn(
        #     "Insufficient data for testing {0} _|_ {1} | {2}. ".format(X, Y, Z)
        #     + "At least {0} samples recommended, {1} present.".format(
        #         5 * num_params, len(data)
        #     )
        # )
        chi2 = 1
        p_value = 1
    else:
        # compute actual frequency/state_count table:
        # = P(X,Y,Zs)
        XYZ_state_counts = pd.crosstab(
            index=data[X], columns=[data[Y]] + [data[z] for z in Z]
        )
        # reindex to add missing rows & columns (if some values don't appear in data)
        row_index = state_names[X]
        column_index = pd.MultiIndex.from_product(
            [state_names[Y]] + [state_names[z] for z in Z], names=[Y] + Z
        )
        if not isinstance(XYZ_state_counts.columns, pd.MultiIndex):
            XYZ_state_counts.columns = pd.MultiIndex.from_arrays([XYZ_state_counts.columns])
        XYZ_state_counts = XYZ_state_counts.reindex(
            index=row_index, columns=column_index
        ).fillna(0)

        # compute the expected frequency/state_count table if X _|_ Y | Zs:
        # = P(X|Zs)*P(Y|Zs)*P(Zs) = P(X,Zs)*P(Y,Zs)/P(Zs)
        if Z:
            XZ_state_counts = XYZ_state_counts.sum(axis=1, level=Z)  # marginalize out Y
            YZ_state_counts = XYZ_state_counts.sum().unstack(Z)  # marginalize out X
        else:
            XZ_state_counts = XYZ_state_counts.sum(axis=1)

            YZ_state_counts = XYZ_state_counts.sum()
        Z_state_counts = YZ_state_counts.sum()  # marginalize out both

        XYZ_expected = pd.DataFrame(
            index=XYZ_state_counts.index, columns=XYZ_state_counts.columns
        )
        for X_val in XYZ_expected.index:
            if Z:
                for Y_val in XYZ_expected.columns.levels[0]:
                    XYZ_expected.loc[X_val, Y_val] = (
                            XZ_state_counts.loc[X_val]
                            * YZ_state_counts.loc[Y_val]
                            / Z_state_counts
                    ).values
            else:
                for Y_val in XYZ_expected.columns:
                    XYZ_expected.loc[X_val, Y_val] = (
                            XZ_state_counts.loc[X_val]
                            * YZ_state_counts.loc[Y_val]
                            / float(Z_state_counts)
                    )

        observed = XYZ_state_counts.values.flatten()
        expected = XYZ_expected.fillna(0).values.flatten()
        # remove elements where the expected value is 0;
        # this also corrects the degrees of freedom for chisquare
        observed, expected = zip(
            *((o, e) for o, e in zip(observed, expected) if not e == 0)
        )

        chi2, p_value = stats.chisquare(observed, expected)
        # p_value = 1.0 - stats.chi2.cdf(x=chi2, df=num_params)

    if p_value >= alpha:
        dep = (-2.0) - chi2 / num_params
    else:
        dep = 2.0 + chi2 / num_params
    # print(significance_level," chi2 is: ", chi2," num_params is: ",num_params)

    return chi2, num_params, dep, p_value

def getPCD(data, target, alaph, is_discrete):
    number, kVar = np.shape(data)
    max_k = 3
    PCD = []
    ci_number = 0

    # use a list of sepset[] to store a condition set which can make target and the variable condition independence
    # the above-mentioned variable will be remove from CanPCD or PCD
    sepset = [[] for i in range(kVar)]

    while True:
        variDepSet = []
        CanPCD = [i for i in range(kVar) if i != target and i not in PCD]
        CanPCD_temp = CanPCD.copy()

        for vari in CanPCD_temp:
            breakFlag = False
            dep_gp_min = float("inf")
            vari_min = -1

            if len(PCD) >= max_k:
                Plength = max_k
            else:
                Plength = len(PCD)

            for j in range(Plength+1):
                SSubsets = subsets(PCD, j)
                for S in SSubsets:
                    ci_number += 1
                    pval_gp, dep_gp = cond_indep_test(data, target, vari, S, is_discrete)

                    if pval_gp > alaph:
                        vari_min = -1
                        CanPCD.remove(vari)
                        sepset[vari] = [i for i in S]
                        breakFlag = True
                        break
                    elif dep_gp < dep_gp_min:
                        dep_gp_min = dep_gp
                        vari_min = vari

                if breakFlag:
                    break

            # use a list of variDepset to store list, like [variable, its dep]
            if vari_min in CanPCD:
                variDepSet.append([vari_min, dep_gp_min])

        # sort list of variDepSet by dep from max to min
        variDepSet = sorted(variDepSet, key=lambda x: x[1], reverse=True)

        # if variDepset is null ,that meaning PCD will not change
        if variDepSet != []:
            y =variDepSet[0][0]
            PCD.append(y)
            pcd_index = len(PCD)
            breakALLflag = False
            while pcd_index >=0:
                pcd_index -= 1
                x = PCD[pcd_index]
                breakFlagTwo = False

                conditionSetALL = [i for i in PCD if i != x]
                if len(conditionSetALL) >= max_k:
                    Slength = max_k
                else:
                    Slength = len(conditionSetALL)

                for j in range(Slength+1):
                    SSubsets = subsets(conditionSetALL, j)
                    for S in SSubsets:
                        ci_number += 1
                        pval_sp, dep_sp = cond_indep_test(data, target, x, S, is_discrete)

                        if pval_sp > alaph:

                            PCD.remove(x)
                            if x == y:
                                breakALLflag = True

                            sepset[x] = [i for i in S]
                            breakFlagTwo = True
                            break
                    if breakFlagTwo:
                        break

                if breakALLflag:
                    break
        else:
            break
    return list(set(PCD)), sepset, ci_number
