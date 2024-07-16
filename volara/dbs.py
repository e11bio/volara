from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Optional, Union

from funlib.persistence.graphs import PgSQLGraphDatabase, SQLiteGraphDataBase
from funlib.persistence.types import Vec

from .utils import StrictBaseModel


class DB(ABC, StrictBaseModel):
    raw_channels: Optional[int] = None
    enhanced_channels: Optional[int] = None
    embedding_channels: Optional[int] = None

    @property
    def default_node_attrs(self) -> dict[str, Union[Any]]:
        return {"position": Vec(float, 3), "size": int, "filtered": bool}

    @property
    def default_edge_attrs(self):
        return {"adj_weight": float, "lr_weight": float, "distance": float}

    @abstractmethod
    def db(self):
        pass

    def init(self):
        try:
            self.db("r")
        except RuntimeError:
            self.db("w")

    @abstractmethod
    def drop(self):
        pass


class SQLite(DB):
    db_type: Literal["sqlite"] = "sqlite"
    path: Path
    node_attrs: Optional[dict[str, Union[str, int]]] = None
    edge_attrs: Optional[dict[str, Union[str, int]]] = None

    def db(self, mode="r"):
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

        return SQLiteGraphDataBase(
            self.path,
            position_attribute="position",
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            mode=mode,
        )

    def drop(self):
        self.path.unlink()


class PostgreSQL(DB):
    db_type: Literal["postgresql"] = "postgresql"
    host: Optional[str] = None
    name: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    node_attrs: Optional[dict[str, Union[str, int]]] = None
    edge_attrs: Optional[dict[str, Union[str, int]]] = None

    def db(self, mode="r"):
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
        db = self.db("w")
        db._drop_tables()
