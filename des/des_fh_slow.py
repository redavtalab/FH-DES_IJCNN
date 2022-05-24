
import numpy as np
import matplotlib.pyplot as plt
from deslib.des.base import BaseDES
from deslib.util.fuzzy_hyperbox import Hyperbox


class DESFH_Slow(BaseDES):

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
                 mis_sample_based=True):
        self.theta = theta
        self.mu = mu
        self.mis_sample_based = mis_sample_based
        self.HBoxes = []
        self.NO_hypeboxes = 0

        ############### it should be based on Clustering #############################
        super(DESFH_Slow, self).__init__(pool_classifiers=pool_classifiers,
                                    with_IH=with_IH,
                                    safe_k=safe_k,
                                    IH_rate=IH_rate,
                                    mode='hybrid',  # hybrid,weighting
                                    random_state=random_state,
                                    DSEL_perc=DSEL_perc)

    #       super(DESFH_Slow, self).__init__(pool_classifiers, k,
    #                                     DFP=DFP,
    #                                     with_IH=with_IH,
    #                                     safe_k=safe_k,
    #                                     IH_rate=IH_rate,
    #                                     mode='weighting',
    #                                     random_state=random_state,
    #                                     DSEL_perc=DSEL_perc)
    #
    #                                    (pool_classifiers=pool_classifiers,
    #                                            with_IH=with_IH,
    #                                            safe_k=safe_k,
    #                                            IH_rate=IH_rate,
    #                                            random_state=random_state,
    #                                            DSEL_perc=DSEL_perc)
    # bb = HyperBoxes

    def fit(self, X, y):

        super(DESFH_Slow, self).fit(X, y)
        if self.mu > 1 or self.mu <= 0:
            raise Exception("The value of Mu must be between 0 and 1.")
        if self.theta > 1 or self.theta <= 0:
            raise Exception("The value of Mu must be between 0 and 1.")

        if self.mis_sample_based == True:
            for classifier_index in range(self.n_classifiers_):
                MissSet_indexes = ~self.DSEL_processed_[:, classifier_index]
                self.setup_hyperboxs(MissSet_indexes, classifier_index)
        else:
            for classifier_index in range(self.n_classifiers_):
                WellSet_indexes = self.DSEL_processed_[:, classifier_index]
                self.setup_hyperboxs(WellSet_indexes, classifier_index)

    def estimate_competence(self, query, neighbors=None, distances=None,
                            predictions=None):

        competences_ = np.zeros([len(query), self.n_classifiers_])
        for index, sample in enumerate(query):
            listboxComp = []

            if len(self.HBoxes) < 1:  # when there is no box, competence is 1 for all classifiers
                competences_ += 1
                break;

            currentClr = self.HBoxes[0].clsr
            clr = 0
            for box in self.HBoxes:
                clr = box.clsr
                if clr != currentClr:
                    listboxComp.sort()

                    if len(listboxComp) > 1:
                        competences_[index, currentClr] = (0.7 * listboxComp[-1] + 0.3 * listboxComp[
                            -2])  # + 8 * listboxComp[-3] + 7 * listboxComp[-4])/34
                    else:
                        competences_[index, currentClr] = listboxComp[-1]  # np.average(listboxComp)
                    currentClr = clr
                    listboxComp = []
                comp = box.membership(sample)
                listboxComp.append(comp)
            listboxComp.sort()
            if len(listboxComp) > 1:
                competences_[index, clr] = (0.7 * listboxComp[-1] + 0.3 * listboxComp[
                    -2])  # ( 10 * listboxComp[-1] + 9 * listboxComp[-2] + 8 * listboxComp[-3] + 7 * listboxComp[-4])/34
            else:
                competences_[index, clr] = listboxComp[-1]  # np.average(listboxComp)

        # competences_=( np.sqrt(len(sample)) - (competences_ * len(box.samples)) )
        competences_ = np.sqrt(len(sample)) - competences_

        #        competences_ = np.sum(self.DSEL_processed_[neighbors, :], axis=1,
        #                             dtype=np.float)
        #        competences_ = competences_ * 0;
        #        competences_[:,0] = 7
        #        competences_[:,1] = 1
        #        competences_[:,2] = 0.2
        #        competences_[:,3] = 0.3
        #        competences_[:,4] = 4

        return competences_

    def setup_hyperboxs(self, samples_ind, classifier):
        #        print(np.size(samples_ind))
        if np.size(samples_ind) < 1:
            pass
        boxes = []
        selected_samples = self.DSEL_data_[samples_ind, :]
        for X in selected_samples:
            # Creation first box
            if len(boxes) < 1:
                # Create the first Box
                b = Hyperbox(v=X, w=X, classifier=classifier, theta=self.theta)
                self.NO_hypeboxes += 1
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
            self.NO_hypeboxes += 1

        self.HBoxes.extend(boxes)

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
        if self.mis_sample_based:
            max_value = np.max(competences, axis=1)
            selected_classifiers = (
                    competences >= self.mu * max_value.reshape(competences.shape[0], -1))
        else:
            min_value = np.min(competences, axis=1)
            selected_classifiers = (
                    self.mu * competences <= min_value.reshape(competences.shape[0], -1))

        return selected_classifiers

