"""evaluation metrics including hit ratio(HR) and NDCG"""

import math
import pandas as pd

class MetronAtK(object):
    def __init__(self, top_k):
        self._top_k= top_k
        self._subjects= None    # subjects which we ran evaluation on

    # ! getter and setter for same value
    # ! if print(me.top_k), then the value of the top_k would be returned
    # ! if me.top_k= 10, then the value of the top_k would become 10
    # ! This is mainly for future access, preventing error by changing some values or wrong references.
    @property       # ! getter function for 'top_k'
    def top_k(self):
        return self._top_k

    @top_k.setter   # ! setter function for 'top_k'
    def top_k(self, top_k):
        self._top_k= top_k

    @property
    def subjects(self):
        return self._subjects

    # ! FOR WHAT?
    @subjects.setter
    def subjects(self, subjects):
        """
        args:
            subjects: list, [test_users, test_items, test_scores, negative users, negative items, negative scores]
        """
        # ! check the data type of subjects
        assert isinstance(subjects, list)
        test_users, test_items, test_scores= subjects[0], subjects[1], subjects[2]
        neg_users, neg_items, neg_scores= subjects[3], subjects[4], subjects[5]

        # the golden set
        test= pd.DataFrame({'user': test_users,
                            'test_item': test_items,
                            'test_score': test_scores})
        # the full set
        full= pd.DataFrame({'user': neg_users+test_users,
                            'item': neg_items+test_items,
                            'score': neg_scores+test_scores})
        full= pd.merge(full, test, on=['user'], how='left')

        # rank theitems according to the scores for each user
        full['rank']= full.groupby('user')['score'].rank(method='first', ascending=False)
        full.sort_values(['user', 'rank'], inplace=True)
        self._subjects= full

    def cal_hit_ratio(self):
        """Hit Ratio @ top k"""
        full, top_k= self._subjects, self._top_k
        top_k= full[full['rank'] <= top_k]
        test_in_top_k= top_k[top_k['test_item'] == top_k['item']]   # golden items hit in the top-k items
        return len(test_in_top_k) * 1.0 / full['user'].nunique()

    def cal_ndcg(self):
        full, top_k= self._subjects, self._top_k
        top_k= full[full['rank'] <= top_k]
        test_in_top_k= top_k[top_k['test_item'] == top_k['item']]
        test_in_top_k['ndcg']= test_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1+x))
        return test_in_top_k['ndcg'].sum() * 1.0 / full['user'].nunique()