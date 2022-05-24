# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from deslib.des.base import BaseDES
from deslib.util.fuzzy_hyperbox import Hyperbox


class DESFHMW(BaseDES):

    def __init__(self, pool_classifiers=None,
                 k=7, DFP=False,
                 with_IH=False,
                 safe_k=None,
                 IH_rate=0.30,
                 random_state=None,
                 knn_classifier='knn',
                 DSEL_perc=0.5,
                 HyperBoxes=[],
                 theta=0.05,
                 mu=0.991,
                ):
        self.theta = theta
        self.mu = mu
        self.HBoxesW = []
        self.NO_hypeboxesW = 0
        self.HBoxesM = []
        self.NO_hypeboxesM = 0

        ############### it should be based on Clustering #############################
        super(DESFHMW, self).__init__(pool_classifiers=pool_classifiers,
                                    with_IH=with_IH,
                                    safe_k=safe_k,
                                    IH_rate=IH_rate,
                                    mode='hybrid',  # hybrid,weighting
                                    random_state=random_state,
                                    DSEL_perc=DSEL_perc)

    def fit(self, X, y):

        super(DESFHMW, self).fit(X, y)
        if self.mu > 1 or self.mu <= 0:
            raise Exception("The value of Mu must be between 0 and 1.")
        if self.theta > 1 or self.theta <= 0:
            raise Exception("The value of Mu must be between 0 and 1.")

        for classifier_index in range(self.n_classifiers_):
            MissSet_indexes = ~self.DSEL_processed_[:, classifier_index]
            self.setup_hyperboxs(MissSet_indexes, classifier_index,type="miss")
        for classifier_index in range(self.n_classifiers_):
            WellSet_indexes = self.DSEL_processed_[:, classifier_index]
            self.setup_hyperboxs(WellSet_indexes, classifier_index,type="well")

###            ####             ###              Memberships
    def estimate_memberships(self,query, HBoxes, mis_sample_based):
        boxes_classifier = np.zeros((len(HBoxes), 1))
        boxes_W = np.zeros((len(HBoxes), self.n_features_))
        boxes_V = np.zeros((len(HBoxes), self.n_features_))
        boxes_center = np.zeros((len(HBoxes), self.n_features_))
        if mis_sample_based:
            class_memberships_ = np.ones([len(query), self.n_classifiers_])
            NO_hypeboxes = self.NO_hypeboxesM
        else:
            class_memberships_ = np.zeros([len(query), self.n_classifiers_])
            NO_hypeboxes = self.NO_hypeboxesW
        for i in range(len(HBoxes)):
            boxes_classifier[i] = HBoxes[i].clsr
            boxes_W[i] = HBoxes[i].Max
            boxes_V[i] = HBoxes[i].Min
            boxes_center[i] = (HBoxes[i].Max + HBoxes[i].Min) / 2
        boxes_W = boxes_W.reshape(NO_hypeboxes, 1, self.n_features_)
        boxes_V = boxes_V.reshape(NO_hypeboxes, 1, self.n_features_)
        boxes_center = boxes_center.reshape(NO_hypeboxes, 1, self.n_features_)
        Xq = query.reshape(1, len(query), self.n_features_)

        ## Membership Calculation
        halfsize = ((boxes_W - boxes_V) / 2).reshape(NO_hypeboxes, 1, self.n_features_)
        d = np.abs(boxes_center - Xq) - halfsize
        d[d < 0] = 0
        dd = np.linalg.norm(d, axis=2)
        dd = dd / np.sqrt(self.n_features_)
        m = 1 - dd  # m: membership
        m = np.power(m, 4)

        classifiers, indices, count = np.unique(boxes_classifier, return_counts=True, return_index=True)
        k = 0
        for clsr in classifiers:
            c_range = range(indices[k], indices[k] + count[k])
            k += 1
            cmat = m[c_range]
            class_memberships_[:, int(clsr)] = np.max(cmat,axis=0)

            # if len(c_range) > 1:
            #     # bb_indexes = np.argpartition(-cmat, kth=2, axis=0)[:2]
            #     bb_indexes = np.argsort(-cmat, axis=0)
            #     b1 = bb_indexes[0, :]
            #     b2 = bb_indexes[1, :]
            #     for i in range(0, len(query)):
            #         class_memberships_[i, int(clsr)] = cmat[b1[i], i] * 0.7 + cmat[b2[i], i] * 0.3
            #     # IndexError: index 2 is out of bounds for axis 0 with size 1
            # else:  # In case that we have only one hyperbox for the class
            #     for i in range(0, len(query)):
            #         class_memberships_[i, int(clsr)] = cmat[0, i]

        #################################################################### mistake ####################################33
        # if mis_sample_based:
        #     class_memberships_ = np.sqrt(self.n_features_) - class_memberships_
        #################################################################### mistake ####################################33
        return class_memberships_
###@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Competence
    def estimate_competence(self, query, neighbors=None, distances=None, predictions=None):
        miss_coefficient = self.estimate_memberships(query, self.HBoxesM, mis_sample_based=True)
        well_coefficient = self.estimate_memberships(query, self.HBoxesW, mis_sample_based=False)
        competence_ = well_coefficient - 2 * miss_coefficient
        return (competence_ + 2 )*7

    def setup_hyperboxs(self, samples_ind, classifier, type):
        #        print(np.size(samples_ind))
        if np.size(samples_ind) < 1:
            pass
        boxes = []
        selected_samples = self.DSEL_data_[samples_ind, :]
        HB_count = 0
        for X in selected_samples:
            # Creation first box
            if len(boxes) < 1:
                # Create the first Box
                b = Hyperbox(v=X, w=X, classifier=classifier, theta=self.theta)
                HB_count += 1
                boxes.append(b)
                continue

            # X is in a box?
            IsInBox = False
            for box in boxes:
                if np.all(box.Min < X) and np.all(box.Max > X):
                    IsInBox = True
            if IsInBox:
                # nop
                continue

            ######################## Expand ############################
            # Finding nearest box
            nDist = np.inf
            nearest_box = None
            for box in boxes:
                dist = np.linalg.norm(X - box.Center);
                if dist < nDist:
                    nearest_box = box
                    nDist = dist
            if nearest_box.is_expandable(X):
                nearest_box.expand(X)
                continue

                ######################## Creation ############################
            #            else:
            b = Hyperbox(v=X, w=X, classifier=classifier, theta=self.theta)
            boxes.append(b)
            HB_count += 1

        if type == "miss":
            self.NO_hypeboxesM += HB_count
            self.HBoxesM.extend(boxes)
        else:
            self.NO_hypeboxesW += HB_count
            self.HBoxesW.extend(boxes)

    def select(self, competences):
        """Selects all base classifiers that obtained a local accuracy of 100%
        in the region of competence (i.e., local oracle). In the case that no
        base classifiers obtain 100% accuracy, the size of the region of
        competence is reduced and the search for the local oracle is restarted.
        Notes
        ------
        Instead of re-applying the method several times (reducing the size of
        the region of competence), we compute the number of consecutive correct
        classification of each base classifier starting from the closest
        neighbor to the more distant in the estimate_competence function.
        The number of consecutive correct classification represents the size
        of the region of competence in which the corresponding base classifier
        is an Local Oracle. Then, we select all base classifiers with the
        maximum value for the number of consecutive correct classification.
        This speed up the selection process.
        Parameters
        ----------
        competences : array of shape = [n_samples, n_classifiers]
            Competence level estimated for each base classifier and test
            example.
        Returns
        -------
        selected_classifiers : array of shape = [n_samples, n_classifiers]
            Boolean matrix containing True if the base classifier is selected,
            False otherwise.
        """
        if competences.ndim < 2:
            competences = competences.reshape(1, -1)

        # Checks which was the max value for each sample
        # (i.e., the maximum number of consecutive predictions)

        # Select all base classifiers with the maximum number of
        #  consecutive correct predictions for each sample.
        max_value = np.max(competences, axis=1)
        selected_classifiers = ( competences >= self.mu * max_value.reshape(competences.shape[0], -1))

        # else:
        #     min_value = np.min(competences, axis=1)
        #     selected_classifiers = (
        #             self.mu * competences <= min_value.reshape(competences.shape[0], -1))

        return selected_classifiers

#