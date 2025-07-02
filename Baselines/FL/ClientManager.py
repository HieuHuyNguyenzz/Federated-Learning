from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Optional
import random
import threading
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion


class ClientManager(ClientManager):
    def __init__(self) -> None:
        self.clients: Dict[str, ClientProxy] = {}
        self._cv = threading.Condition()
        self.seed = 0 # cài đặt seed để fix client tham gia mỗi round

    def __len__(self) -> int:
        return len(self.clients)

    def num_available(self) -> int:
        return len(self)

    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        with self._cv:
            return self._cv.wait_for(
                lambda: len(self.clients) >= num_clients, timeout=timeout
            )

    def register(self, client: ClientProxy) -> bool:
        if client.cid in self.clients:
            return False

        self.clients[client.cid] = client
        with self._cv:
            self._cv.notify_all()

        return True

    def unregister(self, client: ClientProxy) -> None:
        if client.cid in self.clients:
            del self.clients[client.cid]

            with self._cv:
                self._cv.notify_all()

    def all(self) -> Dict[str, ClientProxy]:
        return self.clients

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        
        random.seed(self.seed)
        self.seed +=1
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        available_cids = list(self.clients)

        if num_clients == 1:
            sampled_cids = random.sample(available_cids, num_clients)
            return [self.clients[cid] for cid in sampled_cids]

        sampled_cids = random.sample(available_cids, num_clients)
        return [self.clients[cid] for cid in sampled_cids]