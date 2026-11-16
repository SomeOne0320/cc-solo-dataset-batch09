import tempfile
import threading
from collections import OrderedDict
from datetime import datetime

import pytest
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from dataset import connect

from .conftest import TEST_DATA


def test_valid_database_url(db):
    assert db.url


def test_database_url_query_string(db):
    db = connect("sqlite:///:memory:/?cached_statements=1")
    assert "cached_statements" in db.url, db.url


def test_tables(db, table):
    assert "weather" in db.tables, db.tables


def test_contains(db, table):
    assert "weather" in db, db.tables


def test_create_table(db):
    table = db["foo"]
    assert db.has_table(table.table.name)
    assert len(table.table.columns) == 1, table.table.columns
    assert "id" in table.table.c, table.table.c


def test_create_table_no_ids(db):
    if db.is_mysql or db.is_sqlite:
        return
    table = db.create_table("foo_no_id", primary_id=False)
    assert table.table.name == "foo_no_id"
    assert len(table.table.columns) == 0, table.table.columns


def test_create_table_custom_id1(db):
    pid = "string_id"
    table = db.create_table("foo2", pid, db.types.string(255))
    assert db.has_table(table.table.name)
    assert len(table.table.columns) == 1, table.table.columns
    assert pid in table.table.c, table.table.c
    table.insert({pid: "foobar"})
    assert table.find_one(string_id="foobar")[pid] == "foobar"


def test_create_table_custom_id2(db):
    pid = "string_id"
    table = db.create_table("foo3", pid, db.types.string(50))
    assert db.has_table(table.table.name)
    assert len(table.table.columns) == 1, table.table.columns
    assert pid in table.table.c, table.table.c

    table.insert({pid: "foobar"})
    assert table.find_one(string_id="foobar")[pid] == "foobar"


def test_create_table_custom_id3(db):
    pid = "int_id"
    table = db.create_table("foo4", primary_id=pid)
    assert db.has_table(table.table.name)
    assert len(table.table.columns) == 1, table.table.columns
    assert pid in table.table.c, table.table.c

    table.insert({pid: 123})
    table.insert({pid: 124})
    assert table.find_one(int_id=123)[pid] == 123
    assert table.find_one(int_id=124)[pid] == 124
    with pytest.raises(IntegrityError):
        table.insert({pid: 123})
    db.executable.rollback()


def test_create_table_shorthand1(db):
    pid = "int_id"
    table = db.get_table("foo5", pid)
    assert len(table.table.columns) == 1, table.table.columns
    assert pid in table.table.c, table.table.c

    table.insert({"int_id": 123})
    table.insert({"int_id": 124})
    assert table.find_one(int_id=123)["int_id"] == 123
    assert table.find_one(int_id=124)["int_id"] == 124
    with pytest.raises(IntegrityError):
        table.insert({"int_id": 123})
    db.executable.rollback()


def test_create_table_shorthand2(db):
    pid = "string_id"
    table = db.get_table("foo6", primary_id=pid, primary_type=db.types.string(255))
    assert len(table.table.columns) == 1, table.table.columns
    assert pid in table.table.c, table.table.c

    table.insert({"string_id": "foobar"})
    assert table.find_one(string_id="foobar")["string_id"] == "foobar"


def test_with(db, table):
    init_length = len(table)
    with pytest.raises(ValueError), db:
        table.insert(
            {
                "date": datetime(2011, 1, 1),
                "temperature": 1,
                "place": "tmp_place",
            }
        )
        raise ValueError()
    assert len(table) == init_length


def test_invalid_values(db, table):
    if db.is_mysql or db.is_sqlite:
        # SQLite has flexible typing and won't raise on type mismatches;
        # MySQL does implicit type casting.
        return
    with pytest.raises(SQLAlchemyError):
        table.insert({"date": True, "temperature": "wrong_value", "place": "tmp_place"})


def test_load_table(db, table):
    tbl = db.load_table("weather")
    assert tbl.table.name == table.table.name


def test_query(db, table):
    r = db.query("SELECT COUNT(*) AS num FROM weather").next()
    assert r["num"] == len(TEST_DATA), r


def test_table_cache_updates(db):
    tbl1 = db.get_table("people")
    data = OrderedDict([("first_name", "John"), ("last_name", "Smith")])
    tbl1.insert(data)
    data["id"] = 1
    tbl2 = db.get_table("people")
    assert dict(tbl2.all().next()) == dict(data), (tbl2.all().next(), data)


def test_thread_connections_released():
    """Connections should be released after a transaction ends (issue #425)."""
    # Use a file-based SQLite database — in-memory SQLite uses
    # SingletonThreadPool which doesn't support real multi-threading.
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        db = connect(f"sqlite:///{f.name}")
        # Create the table on the main thread to avoid schema races.
        db["thread_test"].insert({"value": 0})

        def insert_in_thread() -> None:
            with db:
                db["thread_test"].insert({"value": 1})

        threads = []
        for _ in range(5):
            t = threading.Thread(target=insert_in_thread)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # After all threads finish and their transactions commit,
        # their connections should have been released. Only the main
        # thread's connection (from the setup insert) may remain.
        assert len(db.connections) <= 1, (
            f"Expected at most 1 connection, got {len(db.connections)}"
        )
        db.close()


@pytest.fixture(scope="function")
def file_db(tmp_path):
    """A database backed by a real SQLite file on disk.

    Using a file (instead of in-memory) allows verifying what was actually
    committed by connecting a second, independent database object.
    """
    db = connect(f"sqlite:///{tmp_path}/savepoint.db")
    yield db
    db.close()


def _fresh_rows(db, table):
    """Read all rows of ``table`` through a brand new connection."""
    fresh = connect(db.url)
    try:
        return list(fresh[table].find())
    finally:
        fresh.close()


def test_transaction_context_default_commits_and_rolls_back(file_db):
    """db.transaction() without savepoint behaves like ``with db:``."""
    db = file_db
    table = db["default_tx"]
    with db.transaction():
        table.insert({"id": 1})
        assert db.in_transaction
    assert not db.in_transaction
    assert [r["id"] for r in _fresh_rows(db, "default_tx")] == [1]

    with pytest.raises(ValueError), db.transaction():
        table.insert({"id": 2})
        raise ValueError()
    assert not db.in_transaction
    assert [r["id"] for r in _fresh_rows(db, "default_tx")] == [1]


def test_transaction_default_nested_rollback_keeps_inner_writes(file_db):
    """Legacy nesting only tracks depth: an inner rollback does not undo.

    This documents the pre-existing default behavior that
    ``transaction(savepoint=True)`` changes.
    """
    db = file_db
    table = db["legacy_nesting"]
    with db:
        table.insert({"id": 1})
        with pytest.raises(RuntimeError), db.transaction():
            table.insert({"id": 2})
            raise RuntimeError()
        # Caught by the outer layer: the inner rollback only decremented
        # the nesting depth, so the inner row is still part of the outer
        # transaction and gets committed with it.
        assert table.count() == 2
    assert [r["id"] for r in _fresh_rows(db, "legacy_nesting")] == [1, 2]


def test_savepoint_rolls_back_single_customer(file_db):
    """A failing customer's changes are withdrawn, the batch continues."""
    db = file_db
    orders = db["orders"]
    batch = [("alice", 100, True), ("bob", 200, False), ("carol", 300, True)]
    with db:
        for customer, amount, valid in batch:
            try:
                with db.transaction(savepoint=True):
                    orders.insert({"customer": customer, "amount": amount})
                    if not valid:
                        raise ValueError("bad customer data")
            except ValueError:
                # The outer transaction is still usable after the
                # independent inner transaction was rolled back.
                assert db.in_transaction
                continue
        assert db.in_transaction
    assert not db.in_transaction

    rows = {r["customer"]: r["amount"] for r in _fresh_rows(db, "orders")}
    assert rows == {"alice": 100, "carol": 300}, rows


def test_savepoint_in_memory(db):
    """Savepoints also work on the default in-memory database."""
    with db:
        db["mem"].insert({"id": 1})
        with pytest.raises(RuntimeError), db.transaction(savepoint=True):
            db["mem"].insert({"id": 2})
            raise RuntimeError()
        assert db.in_transaction
        db["mem"].insert({"id": 3})
    assert not db.in_transaction
    assert [r["id"] for r in db["mem"].find()] == [1, 3]


def test_savepoint_released_layer_is_not_committed(file_db):
    """A cleanly exited savepoint layer is still undone by outer rollback.

    Covers the case where the outer transaction has not written anything
    before entering the inner transaction (SQLite defers BEGIN).
    """
    db = file_db
    with pytest.raises(RuntimeError), db:
        # The outer transaction has not run any statement yet.
        with db.transaction(savepoint=True):
            db["inner_only"].insert({"id": 1, "amount": 50})
        # Normal exit of the inner layer must not commit anything.
        assert db.in_transaction
        raise RuntimeError()
    assert not db.in_transaction

    fresh = connect(db.url)
    try:
        assert "inner_only" not in fresh.tables, fresh.tables
    finally:
        fresh.close()

    # The next operation must not think it is still in the old
    # transaction: writes outside a transaction auto-commit.
    db["after_rollback"].insert({"id": 1})
    fresh = connect(db.url)
    try:
        assert "after_rollback" in fresh.tables
    finally:
        fresh.close()


def test_savepoint_released_layer_undone_with_prior_outer_writes(file_db):
    """Outer rollback withdraws outer and released inner changes alike."""
    db = file_db
    table = db["both_layers"]
    table.insert({"id": 0, "amount": 1})
    with pytest.raises(RuntimeError), db:
        table.insert({"id": 1, "amount": 10})
        with db.transaction(savepoint=True):
            table.insert({"id": 2, "amount": 20})
        raise RuntimeError()
    assert not db.in_transaction
    rows = _fresh_rows(db, "both_layers")
    assert [r["id"] for r in rows] == [0], rows


def test_savepoint_without_outer_transaction(file_db):
    """Without an outer transaction the entry is a regular transaction."""
    db = file_db
    table = db["standalone"]
    table.insert({"id": 1, "amount": 5})

    # Failure: no data written inside may be committed early or remain.
    with pytest.raises(ValueError), db.transaction(savepoint=True):
        table.insert({"id": 2, "amount": 10})
        spy = connect(db.url)
        try:
            # Uncommitted changes must not be visible elsewhere.
            assert spy["standalone"].count() == 1
        finally:
            spy.close()
        raise ValueError()
    assert not db.in_transaction
    assert [r["id"] for r in _fresh_rows(db, "standalone")] == [1]

    # Success: the block commits as a whole.
    with db.transaction(savepoint=True):
        table.insert({"id": 3, "amount": 30})
        table.insert({"id": 4, "amount": 40})
        assert db.in_transaction
    assert not db.in_transaction
    assert [r["id"] for r in _fresh_rows(db, "standalone")] == [1, 3, 4]


def test_savepoint_multi_level_mixed_nesting(file_db):
    """Independent layers can nest and mix with plain inner transactions."""
    db = file_db
    table = db["multi"]
    with db:  # level 1: regular transaction
        table.insert({"id": 1})
        with db.transaction(savepoint=True):  # level 2: savepoint
            table.insert({"id": 2})
            with db:  # level 3: plain nested transaction (depth only)
                table.insert({"id": 3})
                with db.transaction(savepoint=True):  # level 4: savepoint
                    table.insert({"id": 4})
                    try:
                        with db.transaction(savepoint=True):  # level 5
                            table.insert({"id": 5})
                            raise RuntimeError()
                    except RuntimeError:
                        pass
                    assert db.in_transaction
                    table.insert({"id": 6})
                # level 4 exits cleanly: ids 4 and 6 confirmed for the layer
            # level 3 exits cleanly (no-op)
            try:
                with db.transaction(savepoint=True):  # level 2b
                    table.insert({"id": 7})
                    raise RuntimeError()
            except RuntimeError:
                pass
        # level 2 exits cleanly: ids 2, 3, 4, 6 confirmed for the layer
        table.insert({"id": 8})
    assert not db.in_transaction
    ids = [r["id"] for r in _fresh_rows(db, "multi")]
    assert ids == [1, 2, 3, 4, 6, 8], ids


def test_savepoint_integrity_error_recovery(file_db):
    """A constraint violation inside the layer only undoes that layer."""
    db = file_db
    table = db["constrained"]
    table.insert({"id": 1, "amount": 10})
    with db:
        table.insert({"id": 2, "amount": 20})
        with pytest.raises(IntegrityError), db.transaction(savepoint=True):
            table.insert({"id": 3, "amount": 30})
            # duplicate primary key
            table.insert({"id": 2, "amount": 999})
        assert db.in_transaction
        table.insert({"id": 4, "amount": 40})
    assert not db.in_transaction
    rows = _fresh_rows(db, "constrained")
    assert [(r["id"], r["amount"]) for r in rows] == [
        (1, 10),
        (2, 20),
        (4, 40),
    ], rows


def test_savepoint_query_before_transaction(file_db):
    """A read before the transaction must not break transaction semantics."""
    db = file_db
    table = db["read_first"]
    table.insert({"id": 1, "amount": 1})
    # Run a query before entering the transaction (autobegin state).
    assert table.count() == 1
    assert not db.in_transaction

    with db:
        with db.transaction(savepoint=True):
            table.insert({"id": 2, "amount": 2})
        # The released inner layer must not have been committed yet.
        spy = connect(db.url)
        try:
            assert spy["read_first"].count() == 1
        finally:
            spy.close()
        table.insert({"id": 3, "amount": 3})
    assert not db.in_transaction
    assert [r["id"] for r in _fresh_rows(db, "read_first")] == [1, 2, 3]

    # The next operation starts a fresh transaction and auto-commits.
    table.insert({"id": 4, "amount": 4})
    assert [r["id"] for r in _fresh_rows(db, "read_first")] == [1, 2, 3, 4]


def test_savepoint_rollback_schema_changes(file_db):
    """Rolled-back tables/columns must not leave stale metadata behind."""
    db = file_db
    existing = db["existing"]
    existing.insert({"id": 1, "name": "keep"})

    with db:
        try:
            with db.transaction(savepoint=True):
                new_table = db["brand_new"]
                new_table.insert({"id": 1, "value": 1})
                existing.insert({"id": 2, "name": "x", "extra": "rolled"})
                assert db.has_table("brand_new")
                assert existing.has_column("extra")
                raise RuntimeError()
        except RuntimeError:
            pass
        # The database structure was rolled back with the inner layer.
        assert not db.has_table("brand_new")
        assert not existing.has_column("extra")
        # Already obtained table objects follow the actual structure:
        # the same object re-creates the table and the column is re-added.
        new_table.insert({"id": 5, "value": 5})
        existing.insert({"id": 3, "name": "y", "extra": "kept"})
    assert not db.in_transaction

    fresh = connect(db.url)
    try:
        assert fresh["brand_new"].find_one(id=5)["value"] == 5
        assert fresh["existing"].find_one(id=3)["extra"] == "kept"
        assert fresh["existing"].find_one(id=2) is None
    finally:
        fresh.close()


def test_savepoint_keeps_outer_connection(file_db):
    """Leaving an inner layer must not release the outer connection."""
    db = file_db
    tid = threading.get_ident()
    with db:
        db["conn_check"].insert({"id": 1})
        assert tid in db.connections
        with db.transaction(savepoint=True):
            db["conn_check"].insert({"id": 2})
        # The inner layer ended but the outer transaction still holds
        # its connection and can continue to write.
        assert tid in db.connections
        assert db.in_transaction
        db["conn_check"].insert({"id": 3})
    assert not db.in_transaction
    assert tid not in db.connections
    assert [r["id"] for r in _fresh_rows(db, "conn_check")] == [1, 2, 3]


def test_savepoint_inside_manual_transaction(file_db):
    """The savepoint entry also works inside begin()/commit() blocks."""
    db = file_db
    table = db["manual"]
    db.begin()
    try:
        table.insert({"id": 1})
        with pytest.raises(RuntimeError), db.transaction(savepoint=True):
            table.insert({"id": 2})
            raise RuntimeError()
        assert db.in_transaction
        table.insert({"id": 3})
        db.commit()
    except Exception:
        db.rollback()
        raise
    assert not db.in_transaction
    assert [r["id"] for r in _fresh_rows(db, "manual")] == [1, 3]


def test_savepoint_thread_state_independent(tmp_path):
    """Transaction state is per-thread; inner exits keep outer connections."""
    db = connect(f"sqlite:///{tmp_path}/threads.db")
    db["shared"].insert({"id": 0, "origin": "main"})

    entered = threading.Event()
    checked = threading.Event()
    errors = []

    def worker() -> None:
        try:
            with db:
                db["shared"].insert({"id": 1, "origin": "worker"})
                try:
                    with db.transaction(savepoint=True):
                        db["shared"].insert({"id": 2, "origin": "inner"})
                        entered.set()
                        assert checked.wait(timeout=10)
                        raise RuntimeError("inner fails")
                except RuntimeError:
                    pass
                # The inner layer was rolled back; the outer transaction
                # of this thread is independent and still usable.
                assert db.in_transaction
                db["shared"].insert({"id": 3, "origin": "worker-after"})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=worker)
    t.start()
    try:
        assert entered.wait(timeout=10)
        # The main thread has its own transaction state and connection.
        assert not db.in_transaction
        # WAL mode: the main thread only sees committed data.
        assert {r["id"] for r in db["shared"].find()} == {0}
    finally:
        checked.set()
    t.join(timeout=10)
    assert not t.is_alive()
    assert not errors, errors
    assert not db.in_transaction
    assert {r["id"] for r in db["shared"].find()} == {0, 1, 3}
    db.close()
