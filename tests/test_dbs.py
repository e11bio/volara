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


@pytest.mark.parametrize("port", [None, 5433])
def test_postgresql_port_is_passed_to_provider(port, monkeypatch):
    """``PostgreSQL.port`` reaches funlib's provider; ``None`` (the default) leaves
    the choice to libpq, i.e. ``PGPORT`` or 5432. Needs no live Postgres."""
    captured = {}

    def provider(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("volara.dbs.PgSQLGraphDatabase", provider)
    kwargs = {} if port is None else {"port": port}
    db = PostgreSQL(node_attrs={"color": 3}, ndim=2, **kwargs)
    db.open("r")
    assert captured["db_port"] == port
    assert PostgreSQL.model_validate_json(db.model_dump_json()).port == port


def test_a_sqlite_url_path_is_cast_to_a_local_file(tmp_path):
    """SQLite needs a real file, so a URL `path` maps to a local file under the
    remote root with a warning -- never the relative directory `memory:/...`."""
    from volara.paths import get_remote_root, set_remote_root

    before = get_remote_root()
    set_remote_root(tmp_path / "remote")
    try:
        with pytest.warns(UserWarning, match="SQLite databases live on the local"):
            db = SQLite(
                path="memory://bucket/graph.db",
                node_attrs={"color": 3},
                edge_attrs={"aff": "float"},
            )
            local = db.local
        assert (
            local == (tmp_path / "remote" / "memory" / "bucket" / "graph.db").resolve()
        )
        assert db.id == "graph"
        assert db.model_dump()["path"] == "memory://bucket/graph.db"
        db.open("w")
        assert local.exists()
        db.drop()
        assert not local.exists()
    finally:
        set_remote_root(before)


def test_csr_pipeline_matches_sqlite(bulk):
    """Extract frags and aff agglom into a CSR database give the nodes and
    edges they give in SQLite, although neither task asks for bulk writes."""
    from funlib.geometry import Coordinate

    block, context = Coordinate(10, 10), Coordinate(2, 2)
    sqlite_db, _ = bulk.pipeline("plain", False, block, context)
    csr_db = bulk.db("csr", csr=True)
    bulk.extract_frags(csr_db, "csr", False, block, context)
    bulk.aff_agglom(csr_db, "csr", False, block, context)

    assert bulk.nodes(csr_db) == bulk.nodes(sqlite_db)
    want, got = bulk.edges(sqlite_db), bulk.edges(csr_db)
    assert len(want) > 0 and set(got) == set(want)
    for pair, data in want.items():
        assert got[pair]["aff"] == pytest.approx(data["aff"]), pair


def test_csr_drop_edges_and_drop(bulk):
    from funlib.geometry import Coordinate

    db = bulk.db("csr", csr=True)
    bulk.extract_frags(db, "csr", False, Coordinate(10, 10), Coordinate(2, 2))
    bulk.aff_agglom(db, "csr", False, Coordinate(10, 10), Coordinate(2, 2))
    nodes = bulk.nodes(db)
    db.drop_edges()
    assert bulk.nodes(db) == nodes and not bulk.edges(db)
    db.drop()
    db.drop()  # nothing there: no error, and nothing created
    assert not db.local.exists()
    with pytest.raises(RuntimeError):
        db.open("r")


def test_csr_arrays_are_rebuilt_once_per_task(bulk, monkeypatch):
    """Blocks only append: the arrays are rebuilt when a task ends, not after
    every block, which a worker's blocks would otherwise each trigger."""
    from funlib.geometry import Coordinate
    from funlib.persistence.graphs import CSRGraphDataBase

    builds = []
    build = CSRGraphDataBase._build
    monkeypatch.setattr(
        CSRGraphDataBase,
        "_build",
        lambda self, *a, **k: builds.append(1) or build(self, *a, **k),
    )
    db = bulk.db("csr", csr=True)
    bulk.extract_frags(db, "csr", False, Coordinate(5, 5), Coordinate(2, 2))
    assert len(builds) == 1
    bulk.aff_agglom(db, "csr", False, Coordinate(5, 5), Coordinate(2, 2))
    assert len(builds) == 2
