from contextlib import ExitStack
from itertools import product
from typing import TYPE_CHECKING

import daisy
import networkx as nx

if TYPE_CHECKING:
    from .blockwise import BlockwiseTask


class Pipeline:
    """
    A class to manage combinations of `BlockwiseTask`s that are grouped
    together in a pipeline.
    """

    task_graph: nx.DiGraph

    def __init__(self, task: "BlockwiseTask | None" = None):
        self.task_graph = nx.DiGraph()
        if task is not None:
            self.task_graph.add_node(task)

    def __add__(self, other: "Pipeline | BlockwiseTask") -> "Pipeline":
        """
        The task or pipeline (`task`) gets run in series after `self`.

        This means that every node in `self` without outgoing edges
        gets an edge to all nodes in `task` without incoming edges.
        """
        from volara.blockwise import BlockwiseTask

        if isinstance(other, BlockwiseTask):
            other = Pipeline(other)
        sink_nodes = [
            task
            for task in self.task_graph.nodes()
            if self.task_graph.out_degree(task) == 0
        ]
        source_nodes = [
            task
            for task in other.task_graph.nodes()
            if other.task_graph.in_degree(task) == 0
        ]

        combined_pipeline = Pipeline()
        combined_graph = combined_pipeline.task_graph
        combined_graph.add_nodes_from(self.task_graph.nodes())
        combined_graph.add_edges_from(self.task_graph.edges())
        combined_graph.add_nodes_from(other.task_graph.nodes())
        combined_graph.add_edges_from(other.task_graph.edges())
        for sink, source in product(sink_nodes, source_nodes):
            combined_graph.add_edge(sink, source)

        return combined_pipeline

    def __or__(self, other: "Pipeline | BlockwiseTask") -> "Pipeline":
        """
        The task or pipeline (`task`) gets run in parallel with `self`.

        Task graphs are merged, but no edges are added.
        """
        from volara.blockwise import BlockwiseTask

        if isinstance(other, BlockwiseTask):
            other = Pipeline(other)

        combined_pipeline = Pipeline()
        combined_graph = combined_pipeline.task_graph
        combined_graph.add_nodes_from(self.task_graph.nodes())
        combined_graph.add_edges_from(self.task_graph.edges())
        combined_graph.add_nodes_from(other.task_graph.nodes())
        combined_graph.add_edges_from(other.task_graph.edges())

        return combined_pipeline

    def run_blockwise(self, multiprocessing: bool = True):
        with ExitStack() as stack:
            node_ordering: list[BlockwiseTask] = list(
                nx.topological_sort(self.task_graph)
            )

            task_map: dict[BlockwiseTask, daisy.Task] = {}
            sinks = []
            for node in node_ordering:
                upstream_tasks = [
                    task_map[upstream]
                    for upstream in self.task_graph.predecessors(node)
                ]
                task = node.task(
                    upstream_tasks=upstream_tasks, multiprocessing=multiprocessing
                )
                task = stack.enter_context(task)
                task_map[node] = task
                if self.task_graph.out_degree(node) == 0:
                    sinks.append(task)

            all_tasks = list(task_map.values())

            if multiprocessing:
                daisy.run_blockwise(all_tasks)
            else:
                server = daisy.SerialServer()
                cl_monitor = daisy.cl_monitor.CLMonitor(server)  # noqa
                server.run_blockwise(all_tasks)

    def drop(self):
        for task in self.task_graph.nodes():
            task.drop()
