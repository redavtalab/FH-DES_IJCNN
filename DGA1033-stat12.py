"""
====================================================================
Dynamic selection with linear classifiers: Statistical Experiment
====================================================================
"""

import pickle
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from deslib.dcs import LCA
from deslib.dcs import MLA
from deslib.dcs import OLA
from deslib.dcs import MCB
from deslib.dcs import Rank

from deslib.des import DESKNN
from deslib.des import KNORAE
from deslib.des import KNORAU
from deslib.des import KNOP
from deslib.des import METADES
from deslib.des import DESFH
from deslib.static.oracle import Oracle
from deslib.static.single_best import SingleBest
from deslib.util.datasets import make_P2

import sklearn.preprocessing as preprocessing
import scipy.io as sio
import time
import os
import warnings
import math
from myfunctions import *

warnings.filterwarnings("ignore")


#+##############################################################################


# Prepare the DS techniques. Changing k value to 7.
def initialize_ds(pool_classifiers, uncalibratedpool, X_DSEL, y_DSEL, k=7):
    knorau = KNORAU(pool_classifiers, k=k)
    kne = KNORAE(pool_classifiers, k=k)
    desknn = DESKNN(pool_classifiers, k=k)
    ola = OLA(pool_classifiers, k=k)
    # lca = LCA(pool_classifiers, k=k)
    # mla = MLA(pool_classifiers, k=k)
    mcb = MCB(pool_classifiers, k=k)
    rank = Rank(pool_classifiers, k=k)
    knop = KNOP(pool_classifiers, k=k)
    meta = METADES(pool_classifiers, k=k)
    desfh_w = DESFH(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=False)
    desfh_m = DESFH(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True)
    oracle = Oracle(pool_classifiers)
    single_best = SingleBest(pool_classifiers,n_jobs=-1)
    majority_voting = pool_classifiers

    # UC_knorau = KNORAU(uncalibratedpool, k=k)
    # UC_kne = KNORAE(uncalibratedpool, k=k)
    # UC_desknn = DESKNN(uncalibratedpool, k=k)
    # UC_ola = OLA(uncalibratedpool, k=k)
    # UC_lca = LCA(uncalibratedpool, k=k)
    # UC_mla = MLA(uncalibratedpool, k=k)
    # UC_mcb = MCB(uncalibratedpool, k=k)
    # UC_rank = Rank(uncalibratedpool, k=k)
    # UC_knop = KNOP(uncalibratedpool, k=k)
    # UC_meta = METADES(uncalibratedpool, k=k)
    # UC_desfh_w = DESFH(uncalibratedpool, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=False)
    # UC_desfh_m = DESFH(uncalibratedpool, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True)
    # UC_oracle = Oracle(uncalibratedpool)
    # UC_single_best = SingleBest(uncalibratedpool, n_jobs=-1)
    UC_majority_voting = uncalibratedpool
    list_ds = [majority_voting, single_best, oracle,knorau,ola,desknn,meta,desfh_w,desfh_m]
    methods_names = ['MV', 'SB', 'Oracle', 'KNORA-U', 'OLA', 'DESKNN', 'META-DES', 'FH-DES-C', 'FH-DES-M' ]
    # fit the ds techniques
    for ds in list_ds:
        if ds != majority_voting and ds != UC_majority_voting:
            ds.fit(X_DSEL, y_DSEL)

    return list_ds, methods_names

def write_results_to_file(accuracy,labels,yhat, methods, datasetName):
    path =  "Results-DGA1033/" + datasetName + "Final Results.p"
    rfile = open(path, mode="wb")
    pickle.dump(methods,rfile)
    pickle.dump(accuracy,rfile)
    pickle.dump(labels,rfile)
    pickle.dump(yhat,rfile)
    rfile.close()

def run_process(datasetName):
    redata = sio.loadmat("DataSets/" + datasetName + ".mat")
    data = redata['dataset']
    X = data[:, 0:-1]
    y = data[:, -1]
    print(datasetName, "is readed.")
    state = 0
    print(datasetName, ': ', X.shape)
    ### ### ### ### ### ### ### ### ###
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    X[np.isnan(X)] = 0
    #### **** #### **** #### **** #### **** #### **** #### ****
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(X)
    #### **** #### **** #### **** #### **** #### **** #### ****
    result_one_dataset = np.zeros((NO_techniques, no_itr))
    predicted_labels = np.zeros((NO_techniques, no_itr, math.ceil(len(y)/4)))
    yhat = np.zeros((no_itr, math.ceil(len(y) / 4)))
    for itr in range(0, no_itr):
        if do_train:
            # rand = np.random.randint(1,10000,1)
            rng = np.random.RandomState(state)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y,
                                                                random_state=rng)  # stratify=y
            X_DSEL, X_test, y_DSEL, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test,
                                                              random_state=rng)  # stratify=y_test
            yhat[itr, :] = y_test

            ###########################################################################
            #                               Training                                  #
            ###########################################################################
            learner = Perceptron(max_iter=100, tol=10e-3, alpha=0.001, penalty=None, random_state=rng)
            calibratedmodel = CalibratedClassifierCV(learner, cv=5,method='isotonic')
            # model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, ), random_state=rng)
            uncalibratedpool = BaggingClassifier(learner,n_estimators=NO_classifiers,bootstrap=True,
                                                 max_samples=1.0,
                                                 random_state=rng)
            # uncalibratedpool.fit(X_train, y_train)

            pool_classifiers = BaggingClassifier(calibratedmodel, n_estimators=NO_classifiers, bootstrap=True,
                                                 max_samples=1.0,
                                                 random_state=rng)
            pool_classifiers.fit(X_train,y_train)

            list_ds, methods_names = initialize_ds(pool_classifiers,uncalibratedpool, X_DSEL, y_DSEL, k=7)

            if(save_all_results):
                save_elements(datasetName,itr,state,pool_classifiers,X_train,y_train,X_test,y_test,X_DSEL,y_DSEL,list_ds,methods_names)
        else: # do_not_train
            pool_classifiers,X_train,y_train,X_test,y_test,X_DSEL,y_DSEL,list_ds,methods_names = load_elements(datasetName,itr,state)
        ###########################################################################
        #                               Generalization                            #
        ###########################################################################

        for ind in range(0, len(list_ds)):
            result_one_dataset[ind, itr] = list_ds[ind].score(X_test, y_test) * 100
            if ind==2: # Oracle results --> y should be passed too.
                predicted_labels[ind, itr, :] = list_ds[ind].predict(X_test,y_test)
                continue
            predicted_labels[ind, itr,:] = list_ds[ind].predict(X_test)
        state += 1
    write_results_to_file(result_one_dataset,predicted_labels,yhat, methods_names, datasetName)
    return result_one_dataset,methods_names,list_ds

theta = .27
NO_Hyperbox_Thereshold = 0.99
NO_classifiers =100
no_itr = 20
save_all_results = False
do_train = False
NO_techniques = 9

datasets = {
    #     Data set of DGA1033 report
    # "Audit",
    # "Banana",
    # "Banknote",
    # "Blood",
    # "Breast",
    # "Car",
    # "Datausermodeling",
    # "Faults",
    # "German",
    # "Haberman",
    # "Heart",
    # "ILPD",
    # "Ionosphere",
    # "Laryngeal1",
    # "Laryngeal3",
    # "Lithuanian",
    # "Liver",
    # "Mammographic",
    # "Monk2",
    # "Phoneme",
    # "Pima",
    # "Sonar",
    # "Statlog",
    # "Steel",
    # "Thyroid",
    # "Vehicle",
    # "Vertebral",
    "Voice3",
    # "Weaning",
    "Wine"
# "Iris"
    # "Wholesale",
    #  "Transfusion", low oracle accuracy

    # 30 Dataset
    # Has problem: "Adult", "Glass",  "Ecoli",    "Seeds",         "Voice9"
    # Large: "Magic", "CTG",  "Segmentation", "WDVG1",
}

datasets = sorted(datasets)
list_ds = []
methods_names = []
NO_datasets = len(datasets)
whole_results = np.zeros([NO_datasets,NO_techniques,no_itr])


dataset_count = 0
done_list = []
for datasetName in datasets:
    try:
        result,methods_names,list_ds = run_process(datasetName)
        whole_results[dataset_count,:,:] = result
        dataset_count +=1
        done_list.append(datasetName)

    except:
        print(datasetName, "could not be readed")
        whole_results2 = whole_results[0:len(done_list),:,:]
        write_whole_results_into_excel(whole_results2, done_list, methods_names)
        continue

write_whole_results_into_excel(whole_results, done_list.copy(), methods_names)
path = "Results-DGA1033/WholeResults.p"
rfile = open(path, mode="wb")
pickle.dump(whole_results,rfile)
datasets = done_list
pickle.dump(datasets,rfile)
pickle.dump(methods_names,rfile)
rfile.close()


# pdata = np.concatenate((whole_results[:,0:3 ,:],whole_results[:,10 :14,:],whole_results[:,21:22,:]) , axis=1)
# metName = methods_names[0:3]+ methods_names[10:14] + methods_names[21:22]
# write_in_latex_table(pdata,done_list,metName,rows="datasets")
write_in_latex_table(whole_results,done_list,methods_names,rows="datasets")


duration = 4  # seconds
freq = 440  # Hz
os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
print("STD:" , np.average(np.std(whole_results,2),0))

# methods_names[0:3]+ methods_names[10:14] + methods_names[21:22]