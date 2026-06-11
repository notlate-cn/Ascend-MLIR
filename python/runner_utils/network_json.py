"""Helpers for the network.json schema (see design spec §3.1)."""
import json
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class NetworkJson:
    function: str
    inputs:   List[Dict[str, Any]]
    kernels:  List[Dict[str, Any]]
    outputs:  List[Dict[str, Any]]

    @classmethod
    def load(cls, path) -> "NetworkJson":
        with open(path) as f:
            d = json.load(f)
        return cls(d["function"], d["inputs"], d["kernels"], d["outputs"])

    def ascendc_kernels(self):
        return [k for k in self.kernels if k["kind"] == "ascendc"]

    def aclnn_kernels(self):
        return [k for k in self.kernels if k["kind"] == "aclnn"]

    def kernel_by_id(self, kid):
        for k in self.kernels:
            if k["id"] == kid:
                return k
        raise KeyError(kid)
