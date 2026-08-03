import pytest

from volara.dbs import DB, PostgreSQL, SQLite


def psql_is_available():
    try:
        import psycopg2

        conn = psycopg2.connect(
            dbname="pytest",
        )
        conn.close()
        return True
    except psycopg2.OperationalError:
        return False
    except ImportError:
        return False


@pytest.mark.parametrize(
    "db_type",
    [
        "sqlite",
        pytest.param(
            "postgresql",
            marks=pytest.mark.skipif(
                not psql_is_available(),
                reason="PostgreSQL is not available",
            ),
        ),
    ],
)
def test_dbs(db_type: str, tmp_path):
    db: DB
    if db_type == "sqlite":
        db = SQLite(
            node_attrs={"color": 3},
            edge_attrs={
                "y_aff": "float",
            },
            ndim=2,
            path=tmp_path / "db.sqlite",
        )
    else:
        db = PostgreSQL(
            node_attrs={"color": 3},
            edge_attrs={
                "y_aff": "float",
            },
            ndim=2,
        )

    with pytest.raises(RuntimeError):
        db.open("r")

    db.init()
    graph_provider = db.open("r+")

    g = graph_provider.read_graph()
    assert g.number_of_nodes() == 0
    assert g.number_of_edges() == 0

    g.add_node(
        0,
        position=(0, 0),
        color=[155, 100, 0],
    )
    g.add_node(
        1,
        position=(1, 1),
        color=[55, 100, 155],
    )
    g.add_edge(0, 1, y_aff=0.5)
    graph_provider.write_graph(g)

    graph_provider = db.open("r")
    g2 = graph_provider.read_graph()
    assert g2.number_of_nodes() == 2
    assert g2.number_of_edges() == 1

    assert g2.nodes[0]["color"] == (155, 100, 0)
    assert g2.nodes[1]["color"] == (55, 100, 155)
    assert g2.edges[0, 1]["y_aff"] == 0.5
    assert g2.nodes[0]["position"] == (0, 0)
    assert g2.nodes[1]["position"] == (1, 1)

    db.drop_edges()
    graph_provider = db.open("r")
    g3 = graph_provider.read_graph()

    assert g3.number_of_nodes() == 2
    assert g3.number_of_edges() == 0

    db.drop()
    with pytest.raises(RuntimeError):
        db.open("r")

    graph_provider = db.open("w")


@pytest.mark.parametrize(
    "db_type",
    [
        "sqlite",
        pytest.param(
            "postgresql",
            marks=pytest.mark.skipif(
                not psql_is_available(),
                reason="PostgreSQL is not available",
            ),
        ),
    ],
)
def test_drop_does_not_create(db_type: str, tmp_path):
    """drop()/drop_edges() on a database that was never created must be a no-op and
    must NOT bring the database into existence (regression for PR#33 review: a fresh
    Postgres DB used to be created by drop() via open('w'))."""
    db: DB
    if db_type == "sqlite":
        db = SQLite(
            node_attrs={"color": 3},
            edge_attrs={"y_aff": "float"},
            ndim=2,
            path=tmp_path / "never_created.sqlite",
        )
    else:
        db = PostgreSQL(
            node_attrs={"color": 3},
            edge_attrs={"y_aff": "float"},
            ndim=2,
            name="volara_pytest_never_created",
        )

    # Sanity: the DB does not exist yet.
    with pytest.raises(RuntimeError):
        db.open("r")

    # Dropping a non-existent DB is a successful no-op...
    db.drop_edges()
    db.drop()

    # ...and must not have created it.
    with pytest.raises(RuntimeError):
        db.open("r")


def test_drop_reraises_real_failures(monkeypatch):
    """A real connection failure must NOT be swallowed by drop()/drop_edges().

    Needs no live Postgres: the change is in the ``except`` clause, so we make
    ``open()`` raise. Before the fix both methods caught every ``RuntimeError``,
    so a refused connection was indistinguishable from a fresh database and the
    drop silently "succeeded".

    pytest tests/test_dbs.py::test_drop_reraises_real_failures
    """
    db = PostgreSQL(
        node_attrs={"color": 3},
        edge_attrs={"y_aff": "float"},
        ndim=2,
        name="volara_pytest_never_created",
    )

    def refused(*args, **kwargs):
        raise RuntimeError("could not connect to server: Connection refused")

    monkeypatch.setattr(PostgreSQL, "open", refused)
    with pytest.raises(RuntimeError, match="Connection refused"):
        db.drop()
    with pytest.raises(RuntimeError, match="Connection refused"):
        db.drop_edges()

    # ...while a genuinely fresh database stays a silent no-op.
    def fresh(*args, **kwargs):
        raise RuntimeError("metadata does not exist")

    monkeypatch.setattr(PostgreSQL, "open", fresh)
    db.drop()
    db.drop_edges()
