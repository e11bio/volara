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
