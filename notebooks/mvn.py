# coding=utf-8
#
# The copyright of this file belongs to Feedzai. The file cannot be
# reproduced in whole or in part, stored in a retrieval system,
# transmitted in any form, or by any means electronic, mechanical,
# photocopying, or otherwise, without the prior permission of the owner.
#
# (c) 2022 Feedzai, Strictly Confidential

import numpy as np
import pandas as pd
import warnings

from scipy.stats import multivariate_normal as mvn
from scipy.optimize import fsolve
from typing import Tuple

class TypeIIIBiasSampler():
    def __init__(
        self,
        label_col: str,
        protected_attribute: str,
        recall_first_group: float,
        recall_second_group: float,
        fpr_first_group: float = 0.05,
        fpr_second_group: float = 0.05,
        protected_attribute_values: Tuple[str, str] = None,
        seed: int = 42,
        feature_names: Tuple[str, str] = ("x1", "x2"),
    ):
        self.label_col = label_col
        self.protected_attribute = protected_attribute
        self.recall_first_group = recall_first_group
        self.recall_second_group = recall_second_group
        self.fpr_first_group = fpr_first_group
        self.fpr_second_group = fpr_second_group
        self.pro_attr_values = protected_attribute_values
        self.seed = seed
        self.feature_names = feature_names
        self.mvn_neg, self.mvn_group_1, self.mvn_group_2 = (
            self._calculate_multivariate_normals()
        )

    def __call__(self, data: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        if not inplace:
            data = data.copy()
        if not self.pro_attr_values:
            self.pro_attr_values = data[self.protected_attribute].unique()
            warnings.warn(
                f"Protected attribute values order not passed. Using {self.pro_attr_values}.",
                UserWarning,
            )

        # Add new columns to dataframe
        data[self.feature_names] = 1

        # Conditions for filtering dataframe.
        a_pos = ((data[self.protected_attribute] == self.pro_attr_values[0]) &
                 (data[self.label_col] == 1))
        b_pos = ((data[self.protected_attribute] == self.pro_attr_values[1]) &
                 (data[self.label_col] == 1))
        a_neg = ((data[self.protected_attribute] == self.pro_attr_values[0]) &
                 (data[self.label_col] == 0))
        b_neg = ((data[self.protected_attribute] == self.pro_attr_values[1]) &
                 (data[self.label_col] == 0))

        data.loc[a_pos, self.feature_names] = self.mvn_group_1.rvs(data[a_pos].shape[0])
        data.loc[b_pos, self.feature_names] = self.mvn_group_2.rvs(data[b_pos].shape[0])
        data.loc[a_neg, self.feature_names] = self.mvn_neg.rvs(data[a_neg].shape[0])
        data.loc[b_neg, self.feature_names] = self.mvn_neg.rvs(data[b_neg].shape[0])

        return data

    def _calculate_multivariate_normals(self) -> Tuple[mvn, mvn, mvn]:
        def get_mean(var: Tuple[float, float]) -> Tuple[float, float]:
            intercept, new_mean = var
            new_dist = mvn(mean=[new_mean, 0], cov=cov_matrix)

            obj_1 = 1 - mvn_negative.cdf([intercept, 0]) * 2 - fpr
            obj_2 = 1 - new_dist.cdf([intercept, 0]) * 2 - recall
            return obj_1, obj_2

        cov_matrix = [[1, 0], [0, 1]]
        mvn_negative = mvn(mean=[0, 0], cov=cov_matrix, seed=self.seed)

        fpr_list = [self.fpr_first_group, self.fpr_second_group]
        recall_list = [self.recall_first_group, self.recall_second_group]

        distributions = []
        for fpr, recall in zip(fpr_list, recall_list):
            estimate = [1.0, 1.0]
            _, mean = fsolve(get_mean, estimate)

            np.random.seed(self.seed)
            theta = 0.25 * np.pi
            rotation_matrix = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            mean = np.array([mean, 0])
            rotated_mean = np.matmul(rotation_matrix, mean)
            distributions.append(mvn(mean=rotated_mean, cov=cov_matrix, seed=self.seed))

        return mvn_negative, distributions[0], distributions[1]
    
