from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Optional, Union

from funlib.persistence.graphs import PgSQLGraphDatabase, SQLiteGraphDataBase
from funlib.persistence.types import Vec

from .utils import StrictBaseModel


class DB(ABC, StrictBaseModel):
    """
    A base class for defining the common attributes and methods for all
    database types.
    """
    node_attrs: Optional[dict[str, Union[str, int]]] = None
    edge_attrs: Optional[dict[str, Union[str, int]]] = None

    @property
    def default_node_attrs(self) -> dict[str, Union[Any]]:
        return {"position": Vec(float, 3), "size": int, "filtered": bool}

    @property
    def default_edge_attrs(self):
        return {"distance": float}

    @property
    def graph_attrs(self):
        node_attrs = self.node_attrs if self.node_attrs is not None else {}
        node_attrs = {
            k: (Vec(float, v) if isinstance(v, int) else eval(v))
            for k, v in node_attrs.items()
        }
        node_attrs = {**self.default_node_attrs, **node_attrs}
        edge_attrs = self.edge_attrs if self.edge_attrs is not None else {}
        edge_attrs = {
            k: (Vec(float, v) if isinstance(v, int) else eval(v))
            for k, v in edge_attrs.items()
        }
        edge_attrs = {**self.default_edge_attrs, **edge_attrs}
        return node_attrs, edge_attrs

    @abstractmethod
    def open(self):
        pass

    def init(self):
        try:
            self.open("r")
        except RuntimeError:
            self.open("w")

    @abstractmethod
    def drop(self):
        pass

    @abstractmethod
    def drop_edges(self):
        pass


class SQLite(DB):
    """
    An SQLite database for storing and retrieving graph data.
    """
    db_type: Literal["sqlite"] = "sqlite"
    path: Path

    def open(self, mode="r"):
        node_attrs, edge_attrs = self.graph_attrs

        return SQLiteGraphDataBase(
            self.path,
            position_attribute="position",
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            mode=mode,
        )

    def drop(self):
        if self.path.exists():
            self.path.unlink()
        meta_path = self.path.parent / f"{self.path.stem}-meta.json"
        if meta_path.exists():
            meta_path.unlink()

    def drop_edges(self):
        db = self.open("a")
        db._drop_edges()
        db._create_tables()


class PostgreSQL(DB):
    """
    A PostgreSQL database for storing and retrieving graph data.
    """
    db_type: Literal["postgresql"] = "postgresql"
    host: Optional[str] = None
    name: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None

    def open(self, mode="r"):
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

    def drop(self):
        db = self.open("a")
        db._drop_tables()
        db._create_tables()

    def drop_edges(self):
        db = self.open("a")
        db._drop_edges()
        db._create_tables()
