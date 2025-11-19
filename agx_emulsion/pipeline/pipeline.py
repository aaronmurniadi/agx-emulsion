from abc import ABC, abstractmethod
from typing import List, Any

class PipelineContext:
    def __init__(self, params):
        self.params = params
        self.data = {}

class Node(ABC):
    @abstractmethod
    def process(self, image: Any, context: PipelineContext) -> Any:
        pass

class Pipeline:
    def __init__(self):
        self.nodes: List[Node] = []

    def add_node(self, node: Node):
        self.nodes.append(node)

    def run(self, image: Any, context: PipelineContext) -> Any:
        for node in self.nodes:
            image = node.process(image, context)
        return image
