from abc import ABC, abstractmethod
from typing import List, Any

import time

import numpy as np

class PipelineContext:
    def __init__(self, params):
        self.params = params
        self.data = {}

class Node(ABC):
    supports_chunking = False

    @abstractmethod
    def process(self, image: Any, context: PipelineContext) -> Any:
        pass

class Pipeline:
    def __init__(self):
        self.nodes: List[Node] = []

    def add_node(self, node: Node):
        self.nodes.append(node)

    def run(self, image: Any, context: PipelineContext, progress_callback=None) -> Any:
        chunk_size = context.params.settings.get('chunk_size', 0)
        is_chunked = False
        chunks = []

        with open('runtime.log', 'w') as f:
            f.write("Pipeline Execution Log\n")
            f.write("======================\n")
            
            total_start_time = time.perf_counter()
            
            for i, node in enumerate(self.nodes):
                node_name = node.__class__.__name__
                if progress_callback:
                    progress_callback(node_name, i, len(self.nodes))
                start_time = time.perf_counter()
                
                # Refined logic:
                if chunk_size > 0 and node.supports_chunking:
                    if not is_chunked:
                        # Split
                        original_shape = image.shape
                        h, w = original_shape[:2]
                        chunks = []
                        # Store chunks as (y, x, chunk_data)
                        for i in range(0, h, chunk_size):
                            for j in range(0, w, chunk_size):
                                chunks.append((i, j, image[i:i+chunk_size, j:j+chunk_size]))
                        is_chunked = True
                        image = None
                    
                    # Process
                    new_chunks = []
                    for y, x, chunk in chunks:
                        new_chunks.append((y, x, node.process(chunk, context)))
                    chunks = new_chunks
                    
                else:
                    if is_chunked:
                        # Merge
                        # Determine new shape from chunks
                        # Assuming all chunks have same channels and dtype
                        # and they tile the image.
                        # We need to find the total height and width.
                        # Since we stored (y, x), we can find max y+h and max x+w
                        
                        total_h = 0
                        total_w = 0
                        channels = 0
                        dtype = chunks[0][2].dtype
                        
                        for y, x, chunk in chunks:
                            ch_h, ch_w = chunk.shape[:2]
                            total_h = max(total_h, y + ch_h)
                            total_w = max(total_w, x + ch_w)
                            if len(chunk.shape) > 2:
                                channels = chunk.shape[2]
                        
                        if channels > 0:
                            image = np.zeros((total_h, total_w, channels), dtype=dtype)
                        else:
                            image = np.zeros((total_h, total_w), dtype=dtype)
                            
                        for y, x, chunk in chunks:
                            ch_h, ch_w = chunk.shape[:2]
                            if channels > 0:
                                image[y:y+ch_h, x:x+ch_w, :] = chunk
                            else:
                                image[y:y+ch_h, x:x+ch_w] = chunk
                        
                        is_chunked = False
                        chunks = []
                    
                    image = node.process(image, context)

                end_time = time.perf_counter()
                duration = end_time - start_time
                f.write(f"{node_name}: {duration:.6f} seconds\n")
            
            # Ensure we merge at the end if still chunked
            if is_chunked:
                 # Merge logic duplicated (should be a helper method but inline for now)
                total_h = 0
                total_w = 0
                channels = 0
                dtype = chunks[0][2].dtype
                
                for y, x, chunk in chunks:
                    ch_h, ch_w = chunk.shape[:2]
                    total_h = max(total_h, y + ch_h)
                    total_w = max(total_w, x + ch_w)
                    if len(chunk.shape) > 2:
                        channels = chunk.shape[2]
                
                if channels > 0:
                    image = np.zeros((total_h, total_w, channels), dtype=dtype)
                else:
                    image = np.zeros((total_h, total_w), dtype=dtype)
                    
                for y, x, chunk in chunks:
                    ch_h, ch_w = chunk.shape[:2]
                    if channels > 0:
                        image[y:y+ch_h, x:x+ch_w, :] = chunk
                    else:
                        image[y:y+ch_h, x:x+ch_w] = chunk
                is_chunked = False

            total_end_time = time.perf_counter()
            total_duration = total_end_time - total_start_time
            f.write("----------------------\n")
            f.write(f"Total Execution Time: {total_duration:.6f} seconds\n")
            
        return image
