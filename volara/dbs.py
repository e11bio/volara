from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from funlib.persistence.graphs import PgSQLGraphDatabase, SQLiteGraphDataBase
from funlib.persistence.graphs.graph_database import GraphDataBase
from funlib.persistence.types import Vec

from .utils import StrictBaseModel

AttrsDict = dict[str, Any]


class DB(StrictBaseModel, ABC):
    """
    A base class for defining the common attributes and methods for all
    database types.
    """

    node_attrs: dict[str, str | int] | None = None
    """
    node_attrs defines the custom node attributes you may want to save for
    your super voxels. An example of some custom node attrs is provided here:

        `node_attrs={"raw_intensity": "float"}`

    This adds a float column with name "raw_intensity" to the nodes table
    in your graph db. You can later use this attribute by name to use this
    value in a blockwise task.
    """
    edge_attrs: dict[str, str | int] | None = None
    """
    edge_attrs defines the custom edge attributes you may want to save between
    your super voxels. An example of some custom edge attrs is provided here:

        `edge_attrs={"z_aff": "float", "y_aff": "float", "x_aff": "float"}`

    This adds a float columns with names ["{zyx}_affinity"] to the edges table
    in your graph db. You can later use these attributes by name to use these
    values in a blockwise task.
    """

    ndim: int = 3
    """
    The dimensionality of your spatial data. This is used to define the number
    of coordinates used to index your node locations.
    """

    @property
    def default_node_attrs(self) -> AttrsDict:
        """
        The type definitions for the default node attributes stored in our database.

        Our DBs will always store supervoxels with the following attributes:

        position: `(float, float, float)`:

            The center of mass of the supervoxel

        size: `int`:

            the number of voxels in this supervoxel

        filtered: `bool`:

            whether or not this fragment has been filtered out

        """
        return {"position": Vec(float, self.ndim), "size": int, "filtered": bool}

    @property
    def default_edge_attrs(self) -> AttrsDict:
        """
        The type definitions for the default edge attributes stored in our database.

        Our DBs will always store edges between supervoxels with the following attributes:

        distance: `float`:

            The distance between center of masses of our super voxels

        """
        return {"distance": float}

    @property
    def graph_attrs(self) -> tuple[AttrsDict, AttrsDict]:
        """
        Get all node and edge attributes including default and user provided attributes
        """
        node_attrs = self.node_attrs if self.node_attrs is not None else {}
        parsed_node_attrs = {
            k: (Vec(float, v) if isinstance(v, int) else eval(v))
            for k, v in node_attrs.items()
        }
        parsed_node_attrs = {**self.default_node_attrs, **parsed_node_attrs}
        edge_attrs = self.edge_attrs if self.edge_attrs is not None else {}
        parsed_edge_attrs = {
            k: (Vec(float, v) if isinstance(v, int) else eval(v))
            for k, v in edge_attrs.items()
        }
        parsed_edge_attrs = {**self.default_edge_attrs, **parsed_edge_attrs}
        return parsed_node_attrs, parsed_edge_attrs

    @abstractmethod
    def open(self, mode: str = "r") -> GraphDataBase:
        """
        Return a `funlib.persistence.graphs.GraphDB` instance.
        """
        pass

    def init(self) -> None:
        """
        Create database if it doesn't exist yet.
        """
        try:
            self.open("r")
        except RuntimeError:
            self.open("w")

    @abstractmethod
    def drop(self) -> None:
        """
        Drop all nodes (supervoxels), edges, and metadata
        """
        pass

    @abstractmethod
    def drop_edges(self) -> None:
        """
        Drop all edges between nodes (supervoxels) leaving the nodes and metadata
        in tact.
        """
        pass

    @property
    @abstractmethod
    def id(self) -> str:
        """
        A unique identifier for this databse.
        """
        pass


class SQLite(DB):
    """
    An SQLite database for storing and retrieving graph data.
    """

    db_type: Literal["sqlite"] = "sqlite"
    """
    A literal used for pydantic serialization and deserialization of
    DB Union types.
    """
    path: Path
    """
    The path to the SQLite db file to use.
    """

    @property
    def id(self) -> str:
        return self.path.stem

    def open(self, mode="r") -> SQLiteGraphDataBase:
        node_attrs, edge_attrs = self.graph_attrs

        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True)

        return SQLiteGraphDataBase(
            self.path,
            position_attribute="position",
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            mode=mode,
        )

    def drop(self) -> None:
        if self.path.exists():
            self.path.unlink()
        meta_path = self.path.parent / f"{self.id}-meta.json"
        if meta_path.exists():
            meta_path.unlink()

    def drop_edges(self) -> None:
        try:
            db = self.open("r+")
            db._drop_edges()
            db._create_tables()
        except RuntimeError:
            pass


class PostgreSQL(DB):
    """
    A PostgreSQL database for storing and retrieving graph data.
    """

    db_type: Literal["postgresql"] = "postgresql"
    host: str = "localhost"
    name: str = "volara"
    user: str | None = None
    password: str | None = None

    @property
    def id(self) -> str:
        return self.name

    def open(self, mode="r") -> PgSQLGraphDatabase:
        node_attrs, edge_attrs = self.graph_attrs

        return PgSQLGraphDatabase(
            db_host=self.host,
            db_name=self.name,
            db_user=self.user,
            db_password=self.password,
            position_attribute="position",
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            mode=mode,
        )

    def drop(self) -> None:
        try:
            db = self.open("r+")
            db._drop_tables()
            db._create_tables()
        except RuntimeError:
            # DB doesn't exist yet
            pass

    def drop_edges(self) -> None:
        try:
            db = self.open("r+")
            db._drop_edges()
            db._create_tables()
        except RuntimeError:
            # DB doesn't exist yet
            pass
