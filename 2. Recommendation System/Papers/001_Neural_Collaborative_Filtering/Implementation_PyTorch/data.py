import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

random.seed(0)


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:

            target tensor: corresponding rating for <user, item> pair
        """
        self.user_tensor= user_tensor
        self.item_tensor= item_tensor
        self.target_tensor= target_tensor

    # !we need to override __getitem__() and __len__() for subset of Dataset
    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


class SampleGenerator(object):
    """Construct Dataset for NCF"""

    def __init__(self, ratings):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns - ['userId', 'itemId', 'rating', 'timestamp']
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        self.ratings= ratings

        # if it's explicit feedback, use '_normalize' / else(implicit feedback), use '_binarize'
        self.preprocess_ratings= self._binarize(ratings)

        # !do we have to use '.unique()'? we used set already
        self.user_pool= set(self.ratings['userId'].unique())
        self.item_pool= set(self.ratings['itemId'].unique())

        # create negative item samples for NCF learning
        self.negatives= self._sample_negative(ratings)

        self.train_ratings, self.test_ratings= self._split_loo(self.preprocess_ratings)

    # ! method for preprocessing explicit feedback (like 0~5 ratings)
    # ! isn't the code below just for min-max scaling? is it normalizing?
    def _normalize(self, ratings):
        """normalize into [0, 1] from [0, max_string] for explicit feedback"""
        ratings= deepcopy(ratings)
        max_rating= ratings.rating.max()
        ratings['rating']= ratings.rating * 1.0 / max_rating
        return ratings

    def _binarize(self, ratings):
        """binarize into 0 or 1, implicit feedback"""
        ratings= deepcopy(ratings)
        ratings['rating'][ratings['rating'] > 0] = 1.0
        return ratings

    def _split_loo(self, ratings):
        """leave one out train/test split"""
        # ! after groupby with userId, rank the result by the 'first' timestamp of each userId
        # ! by doing this, we can held out each user's latest interaction as the test set
        ratings['rank_latest']= ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test= ratings[ratings['rank_latest'] == 1]
        train= ratings[ratings['rank_latest'] > 1]

        # ! check if there are same number of users in train-test sets.
        assert train['userId'].nunique() == test['userId'].nunique()
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def _sample_negative(self, ratings):
        """return all negative items & 100 samples negative items"""
        # ! group items with interactions for each users
        interact_status= ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns= {'itemId': 'interacted_items'}
        )
        interact_status['negative_items']= interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples']= interact_status['negative_items'].apply(lambda x: random.sample(x,99))
        return interact_status[['userId', 'negative_items', 'negative_samples']]

    def instance_a_train_loader(self, num_negatives, batch_size):
        """instance train loader for one training epoch"""
        users, items, ratings= [], [], []
        train_ratings= pd.merge(self.test_ratings, self.negatives[['userId', 'negative_items']], on='userId')
        train_ratings['negatives']= train_ratings['negative_items'].apply(lambda x: random.sample(x, num_negatives))

        # ! .itertuples() return tuples with (index, col 1, col 2,  ... col n)                                                                                       ))
        for row in train_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))
            # ! so if num_negatives==2, the ratio b/w positive and negative is 1:2
            for i in range(num_negatives):
                users.append(int(row.userId))
                items.append(int(row.negatives[i]))
                ratings.append(float(0))    # negative samples get 0 rating
        dataset= UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                       item_tensor=torch.LongTensor(items),
                                       target_tensor=torch.FloatTensor(ratings))

        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @property
    def evaluate_data(self):
        """Create Evaluate Data"""
        test_ratings= pd.merge(self.test_ratings, self.negatives[['userId', 'negative_samples']], on='userId')
        test_users, test_items, negative_users, negative_items= [], [], [], []
        for row in test_ratings.itertuples():
            test_users.append(int(row.userId))
            test_items.append(int(row.itemId))
            for i in range(len(row.negative_samples)):
                negative_users.append(int(row.userId))
                negative_items.append(int(row.negative_samples[i]))
        return [torch.LongTensor(test_users), torch.LongTensor(test_items),
                torch.LongTensor(negative_users), torch.LongTensor(negative_items)]