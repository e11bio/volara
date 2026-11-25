import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Literal

from funlib.persistence.graphs import PgSQLGraphDatabase, SQLiteGraphDataBase
from funlib.persistence.graphs.csr_graph_database import CSRGraphDataBase
from funlib.persistence.graphs.sql_graph_database import SQLGraphDataBase
from funlib.persistence.types import Vec

from .paths import PydanticUPath, local_path
from .utils import StrictBaseModel

AttrsDict = dict[str, Any]
GraphProvider = SQLGraphDataBase | CSRGraphDataBase


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

    bulk_only: ClassVar[bool] = False
    """
    True for a database whose writes only become readable when a task ends.
    Tasks write to it through their bulk path whatever their ``bulk_write``.
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
    def open(self, mode: str = "r") -> GraphProvider:
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
    path: PydanticUPath
    """
    The path to the SQLite db file to use. SQLite needs a real file with
    locking, so a URL here is cast to a local path under
    :func:`volara.paths.get_remote_root` with a warning (see
    :func:`volara.paths.local_path`); use :class:`PostgreSQL` for a shared
    remote database.
    """

    @property
    def local(self) -> Path:
        """The local file behind `path`."""
        return local_path(self.path, what="SQLite databases")

    @property
    def id(self) -> str:
        return self.local.stem

    def open(self, mode="r") -> SQLiteGraphDataBase:
        node_attrs, edge_attrs = self.graph_attrs

        if not self.local.parent.exists():
            self.local.parent.mkdir(parents=True)

        return SQLiteGraphDataBase(
            self.local,
            position_attribute="position",
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            mode=mode,
        )

    def drop(self) -> None:
        if self.local.exists():
            self.local.unlink()
        meta_path = self.local.parent / f"{self.id}-meta.json"
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
    port: int | None = None
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
            db_port=self.port,
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
        except RuntimeError as e:
            # A fresh database has no schema/metadata yet, so open('r+') (a read mode)
            # raises "metadata does not exist". There is nothing to drop, so this is a
            # successful no-op -- drop() must NEVER bring a database into existence
            # (DB.init() is the only creator). Re-raise anything else so real failures
            # (e.g. a broken connection) aren't silently swallowed.
            if "metadata does not exist" in str(e):
                return
            raise
        db._drop_tables()
        db._create_tables()

    def drop_edges(self) -> None:
        try:
            db = self.open("r+")
        except RuntimeError as e:
            # Fresh DB (no schema): nothing to drop, no-op. See drop() -- must not create.
            if "metadata does not exist" in str(e):
                return
            raise
        db._drop_edges()
        db._create_tables()


class CSR(DB):
    """
    A graph in memory-mapped CSR arrays under a directory
    (:class:`funlib.persistence.graphs.CSRGraphDataBase`).

    Workers append what they write to their own files and never contend. The
    arrays readers use are rebuilt when a task ends, so a task always writes
    through its bulk path, and what a task wrote cannot be read back until
    that task has finished. Reads cost the same whatever the size of the
    graph. Existing nodes and edges take new attribute values; rows are not
    updated one at a time.
    """

    db_type: Literal["csr"] = "csr"
    """
    A literal used for pydantic serialization and deserialization of
    DB Union types.
    """
    path: PydanticUPath
    """
    The directory of the database. The arrays are memory-mapped, so a URL
    here is cast to a local path under :func:`volara.paths.get_remote_root`
    with a warning (see :func:`volara.paths.local_path`).
    """
    cell_size: float | list[float] = 3000.0
    """
    Edge length of the grid cells nodes are sorted by, in world units. Pick
    it a few times smaller than the blocks that will be read.
    """

    bulk_only: ClassVar[bool] = True

    @property
    def local(self) -> Path:
        """The local directory behind `path`."""
        return local_path(self.path, what="CSR graph databases")

    @property
    def id(self) -> str:
        return self.local.stem

    def _stored(self) -> dict | None:
        """The metadata of the store on disk, or None if there is none."""
        meta = self.local / "meta.json"
        return json.loads(meta.read_text()) if meta.exists() else None

    def open(self, mode="r") -> CSRGraphDataBase:
        """
        Open the database. ``"w"`` creates it with volara's default attributes
        and the ones configured here. A read mode opens whatever is stored:
        the attributes configured here must be in the store, the default ones
        (``size``, ``filtered``, ``distance``) need not be, so a store that
        volara did not write opens too.
        """
        node_attrs, edge_attrs = self.graph_attrs
        stored = self._stored() if mode != "w" else None
        if stored is not None:
            asked_nodes = self.node_attrs or {}
            asked_edges = self.edge_attrs or {}
            node_attrs = {
                name: attr_type
                for name, attr_type in node_attrs.items()
                if name in asked_nodes or name in stored["node_attrs"]
            }
            edge_attrs = {
                name: attr_type
                for name, attr_type in edge_attrs.items()
                if name in asked_edges or name in stored["edge_attrs"]
            }
        if mode == "w" and not self.local.parent.exists():
            self.local.parent.mkdir(parents=True)
        return CSRGraphDataBase(
            self.local,
            position_attribute="position",
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            mode=mode,
            cell_size=self.cell_size,
        )

    def drop(self) -> None:
        if self.local.exists():
            shutil.rmtree(self.local)

    def drop_edges(self) -> None:
        try:
            db = self.open("r+")
        except RuntimeError:
            return
        db.drop_edges()
