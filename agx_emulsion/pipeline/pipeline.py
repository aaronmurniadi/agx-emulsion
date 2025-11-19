from abc import ABC, abstractmethod
from typing import List, Any

import time

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
        with open('runtime.log', 'w') as f:
            f.write("Pipeline Execution Log\n")
            f.write("======================\n")
            
            total_start_time = time.perf_counter()
            
            for node in self.nodes:
                node_name = node.__class__.__name__
                start_time = time.perf_counter()
                image = node.process(image, context)
                end_time = time.perf_counter()
                duration = end_time - start_time
                f.write(f"{node_name}: {duration:.6f} seconds\n")
            
            total_end_time = time.perf_counter()
            total_duration = total_end_time - total_start_time
            f.write("----------------------\n")
            f.write(f"Total Execution Time: {total_duration:.6f} seconds\n")
            
        return image
