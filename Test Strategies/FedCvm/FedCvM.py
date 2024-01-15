from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
from functools import reduce

from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import FedAvg
import numpy as np

class FedImp(FedAvg):

    def aggregate_impurity(self, results: List[Tuple[NDArrays, int, float]]) -> NDArrays:
      sum_CvM = np.sum([CvM for _, _, CvM in results])
      weighted_weights = [
          [layer * CvM for layer in weights] for weights, _, CvM in results
      ]

      weights_prime: NDArrays = [
          reduce(np.add, layer_updates) / sum_CvM
          for layer_updates in zip(*weighted_weights)
      ]
      return weights_prime

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, fit_res.metrics['CvM'])
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(
            self.aggregate_impurity(weights_results)
        )

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated