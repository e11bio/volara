import networkx as nx

from .blockwise import BlockwiseTask


class Pipeline():
    """
    A class to manage combinations of `BlockwiseTask`s that are grouped
    together in a pipeline.
    """
    task_graph: nx.DiGraph()

    def __init__(self, task_graph: nx.DiGraph):
        self.task_graph = task_graph

    def __addr__(self, task: BlockwiseTask | "Pipeline") -> "Pipeline":
        """
        The task or pipeline (`task`) gets run in series after `self`.

        This means that every node in `self` without outgoing edges
        gets an edge to all nodes in `task` without incoming edges.
        """
        pass

    def __addl__(self, task: BlockwiseTask | "Pipeline") -> "Pipeline":
        """
        The task or pipeline (`task`) gets run in series before `self`.

        This means that every node in `task` without outgoing edges
        gets an edge to all nodes in `self` without incoming edges.
        """
        pass

    def __or__(self, task: BlockwiseTask | "Pipeline") -> "Pipeline":
        """
        The task or pipeline (`task`) gets run in parallel with `self`.

        Task graphs are merged, but no edges are added.
        """
        pass

    def __ror__(self, task: BlockwiseTask | "Pipeline") -> "Pipeline":
        """
        The task or pipeline ('task') gets run in parallel with `self`.

        Task graphs are merged, but no edges are added.
        """
        pass

