from tomotwin.modules.inference.locator import Locator
import pandas as pd
from typing import List

class NaiveLocator(Locator):

    def __init__(self, pthresh: float = None, dthresh: float = None):
        self.pthresh = pthresh
        self.dthresh = dthresh

    def locate(self, classify_output : pd.DataFrame) -> List[pd.DataFrame]:
        """

            :param classify_output: Output from the tomotwin_classify
            :return: For each class a separate dataframe
            """
        particles_dataframes = []

        df_particles = classify_output[classify_output["predicted_class"] != -1]

        import numpy as np
        unique_classes = np.unique(df_particles["predicted_class"])
        for id in unique_classes:
            df_id = df_particles[df_particles["predicted_class"] == id]

            if self.pthresh is not None:
                df_id = df_id[df_id["predicted_prob"] > self.pthresh]

            if self.dthresh is not None:
                df_id = df_id[df_id[f"d_class_{id}"] < self.dthresh]
            particles_dataframes.append(df_id)
        return particles_dataframes

