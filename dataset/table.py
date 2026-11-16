import logging
import threading
import warnings
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal

from sqlalchemy import false, func, select
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.schema import Column, Index
from sqlalchemy.schema import Table as SQLATable
from sqlalchemy.sql import and_, expression, union_all
from sqlalchemy.sql.expression import (
    ClauseElement,
    ColumnElement,
    UnaryExpression,
    bindparam,
)

from dataset.types import MYSQL_LENGTH_TYPES, ColumnType, Types
from dataset.util import (
    QUERY_STEP,
    DatasetError,
    MutableRow,
    OutRow,
    QueryError,
    ResultIter,
    SQLWriteValue,
    WriteRow,
    ensure_strings,
    index_name,
    normalize_column_key,
    normalize_column_name,
    normalize_table_name,
    pad_chunk_columns,
)

if TYPE_CHECKING:
    from dataset.database import Database

log = logging.getLogger(__name__)


class Table:
    """Represents a table in a database and exposes common operations."""

    PRIMARY_DEFAULT = "id"

    def __init__(
        self,
        database: "Database",
        table_name: str,
        primary_id: str | Literal[False] | None = None,
        primary_type: ColumnType | None = None,
        primary_increment: bool | None = None,
        auto_create: bool = False,
    ):
        """Initialise the table from database schema."""
        self.db = database
        self.name = normalize_table_name(table_name)
        self._table: SQLATable | None = None
        self._columns: dict[str, str] | None = None
        self._indexes: list[set[str]] = []
        self._primary_id: str | Literal[False] = (
            primary_id if primary_id is not None else self.PRIMARY_DEFAULT
        )
        self._primary_type: ColumnType = (
            primary_type if primary_type is not None else Types.integer
        )
        if primary_increment is None:
            primary_increment = self._primary_type in (Types.integer, Types.bigint)
        self._primary_increment = primary_increment
        self._auto_create = auto_create

    @property
    def exists(self) -> bool:
        """Check to see if the table currently exists in the database."""
        if self._table is not None:
            return True
        return self.name in self.db

    @property
    def table(self) -> SQLATable:
        """Get a reference to the table, which may be reflected or created.

        This property guarantees to return a non-None SQLATable instance.
        If the table doesn't exist and auto_create is False, raises DatasetError.
        """
        if self._table is None:
            self._sync_table(())
        assert self._table is not None, "_sync_table should ensure _table is set"
        return self._table

    @property
    def _column_keys(self) -> dict[str, str]:
        """Get a dictionary of all columns and their case mapping."""
        if not self.exists:
            return {}
        with self.db.lock:
            if self._columns is None:
                # Initialise the table if it doesn't exist
                table = self.table
                self._columns = {}
                for column in table.columns:
                    name = normalize_column_name(column.name)
                    key = normalize_column_key(name)
                    if key in self._columns:
                        log.warning("Duplicate column: %s", name)
                    if key is None:
                        log.warning("Invalid column name: %s", name)
                        continue
                    self._columns[key] = name
            return self._columns

    @property
    def columns(self) -> list[str]:
        """Get a listing of all columns that exist in the table."""
        return list(self._column_keys.values())

    def has_column(self, column: str | None) -> bool:
        """Check if a column with the given name exists on this table."""
        if column is None:
            return False
        key = normalize_column_key(normalize_column_name(column))
        return key in self._column_keys

    def _get_column_name(self, name: str) -> str:
        """Find the best column name with case-insensitive matching."""
        name = normalize_column_name(name)
        key = normalize_column_key(name)
        if key is None:
            return name
        return self._column_keys.get(key, name)

    def insert(
        self,
        row: WriteRow,
        ensure: bool | None = None,
        types: dict[str, ColumnType] | None = None,
    ) -> Any:
        """Add a ``row`` dict by inserting it into the table.

        If ``ensure`` is set, any of the keys of the row are not
        table columns, they will be created automatically.

        During column creation, ``types`` will be checked for a key
        matching the name of a column to be created, and the given
        SQLAlchemy column type will be used. Otherwise, the type is
        guessed from the row value, defaulting to a simple unicode
        field.
        ::

            data = dict(title='I am a banana!')
            table.insert(data)

        Returns the inserted row's primary key.
        """
        row = self._sync_columns(row, ensure, types=types)
        res = self.db.executable.execute(self.table.insert().values(row))
        self.db._auto_commit()
        if res.inserted_primary_key is not None and len(res.inserted_primary_key) > 0:
            return res.inserted_primary_key[0]
        return True

    def insert_ignore(
        self,
        row: WriteRow,
        keys: Sequence[str],
        ensure: bool | None = None,
        types: dict[str, ColumnType] | None = None,
    ) -> Any:
        """Add a ``row`` dict into the table if the row does not exist.

        If rows with matching ``keys`` exist no change is made.

        Setting ``ensure`` results in automatically creating missing columns,
        i.e., keys of the row are not table columns.

        During column creation, ``types`` will be checked for a key
        matching the name of a column to be created, and the given
        SQLAlchemy column type will be used. Otherwise, the type is
        guessed from the row value, defaulting to a simple unicode
        field.
        ::

            data = dict(id=10, title='I am a banana!')
            table.insert_ignore(data, ['id'])
        """
        row = self._sync_columns(row, ensure, types=types)
        if self._check_ensure(ensure):
            self.create_index(keys)
        args, _ = self._keys_to_args(row, keys)
        if self.count(**args) == 0:
            return self.insert(row, ensure=False)
        return False

    def insert_many(
        self,
        rows: Sequence[WriteRow],
        chunk_size: int = 1000,
        ensure: bool | None = None,
        types: dict[str, ColumnType] | None = None,
    ) -> None:
        """Add many rows at a time.

        This is significantly faster than adding them one by one. Per default
        the rows are processed in chunks of 1000 per commit, unless you specify
        a different ``chunk_size``.

        See :py:meth:`insert() <dataset.Table.insert>` for details on
        the other parameters.
        ::

            rows = [dict(name='Dolly')] * 10000
            table.insert_many(rows)
        """
        # Sync table before inputting rows.
        sync_row: MutableRow = {}
        for row in rows:
            # Only get non-existing columns.
            sync_keys = list(sync_row.keys())
            for key in [k for k in row if k not in sync_keys]:
                # Get a sample of the new column(s) from the row.
                sync_row[key] = row[key]
        self._sync_columns(sync_row, ensure, types=types)

        # Get columns name list to be used for padding later.
        columns = sync_row.keys()

        chunk: list[MutableRow] = []
        for index, row in enumerate(rows):
            chunk.append(dict(row))

            # Insert when chunk_size is fulfilled or this is the last row
            if len(chunk) == chunk_size or index == len(rows) - 1:
                chunk = pad_chunk_columns(chunk, columns)
                self.db.executable.execute(self.table.insert(), chunk)
                self.db._auto_commit()
                chunk = []

    def update(
        self,
        row: WriteRow,
        keys: Sequence[str],
        ensure: bool | None = None,
        types: dict[str, ColumnType] | None = None,
        return_count: bool = False,
    ) -> bool | int:
        """Update a row in the table.

        The update is managed via the set of column names stated in ``keys``:
        they will be used as filters for the data to be updated, using the
        values in ``row``.
        ::

            # update all entries with id matching 10, setting their title
            # columns
            data = dict(id=10, title='I am a banana!')
            table.update(data, ['id'])

        If keys in ``row`` update columns not present in the table, they will
        be created based on the settings of ``ensure`` and ``types``, matching
        the behavior of :py:meth:`insert() <dataset.Table.insert>`.
        """
        row = self._sync_columns(row, ensure, types=types)
        args, row = self._keys_to_args(row, keys)
        clause = self._args_to_clause(args)
        if not len(row):
            return self.count(clause)
        stmt = self.table.update().where(clause).values(row)
        rp = self.db.executable.execute(stmt)
        self.db._auto_commit()
        if rp.supports_sane_rowcount():
            return rp.rowcount
        if return_count:
            return self.count(clause)
        return False

    def update_many(
        self,
        rows: Sequence[WriteRow],
        keys: Sequence[str],
        chunk_size: int = 1000,
        ensure: bool | None = None,
        types: dict[str, ColumnType] | None = None,
    ) -> None:
        """Update many rows in the table at a time.

        This is significantly faster than updating them one by one. Per default
        the rows are processed in chunks of 1000 per commit, unless you specify
        a different ``chunk_size``.

        See :py:meth:`update() <dataset.Table.update>` for details on
        the other parameters.
        """
        keys = ensure_strings(keys)

        chunk: list[MutableRow] = []
        columns: list[str] = []
        for index, row in enumerate(rows):
            columns.extend(
                col for col in row if (col not in columns) and (col not in keys)
            )

            # bindparam requires names to not conflict (cannot be "id" for id)
            row_ = dict(row)
            for key in keys:
                row_[f"_{key}"] = row_[key]
                row_.pop(key)
            chunk.append(row_)

            # Update when chunk_size is fulfilled or this is the last row
            if len(chunk) == chunk_size or index == len(rows) - 1:
                cl = [self.table.c[k] == bindparam(f"_{k}") for k in keys]
                stmt = (
                    self.table.update()
                    .where(and_(True, *cl))
                    .values({col: bindparam(col, required=False) for col in columns})
                )
                self.db.executable.execute(stmt, chunk)
                self.db._auto_commit()
                chunk = []

    def upsert(
        self,
        row: WriteRow,
        keys: Sequence[str],
        ensure: bool | None = None,
        types: dict[str, ColumnType] | None = None,
    ) -> Any:
        """An UPSERT is a smart combination of insert and update.

        If rows with matching ``keys`` exist they will be updated, otherwise a
        new row is inserted in the table.
        ::

            data = dict(id=10, title='I am a banana!')
            table.upsert(data, ['id'])
        """
        row = self._sync_columns(row, ensure, types=types)
        if self._check_ensure(ensure):
            self.create_index(keys)
        row_count = self.update(row, keys, ensure=False, return_count=True)
        if row_count == 0:
            return self.insert(row, ensure=False)
        return True

    def upsert_many(
        self,
        rows: Sequence[WriteRow],
        keys: Sequence[str],
        chunk_size: int = 1000,
        ensure: bool | None = None,
        types: dict[str, ColumnType] | None = None,
    ) -> None:
        """
        Sorts multiple input rows into upserts and inserts. Inserts are passed
        to insert and upserts are updated.

        Rows are processed in chunks of ``chunk_size``. Within each chunk,
        rows sharing the same key are coalesced in input order and the
        existence of the keys is established with a single query over just
        the keys in that chunk. Existing keys are updated in grouped
        statements and new keys are inserted in grouped statements, rather
        than issuing a SELECT and write for every row.

        See :py:meth:`upsert() <dataset.Table.upsert>` and
        :py:meth:`insert_many() <dataset.Table.insert_many>`.
        """
        row_list = list(rows)
        if not row_list:
            return
        key_names = ensure_strings(keys)

        # Create every column used by these rows before any write, matching
        # the first-seen sample and type hints used by per-row writes. Keys
        # are deduplicated the same way ``_get_column_name`` resolves them, so
        # differently-cased spellings of a new column create it only once.
        sync_row: MutableRow = {}
        sync_keys: set[str] = set()
        for row in row_list:
            for name, value in row.items():
                key = normalize_column_key(normalize_column_name(name))
                assert key is not None
                if key not in sync_keys:
                    sync_keys.add(key)
                    sync_row[name] = value
        self._sync_columns(sync_row, ensure, types=types)

        if self._check_ensure(ensure):
            self.create_index(key_names)

        chunk_size = max(1, chunk_size)
        for start in range(0, len(row_list), chunk_size):
            chunk = row_list[start : start + chunk_size]
            self._upsert_many_chunk(chunk, key_names, types)

    # Maximum number of key tuples to include in one compound SELECT. SQLite
    # limits compound SELECTs to 500 terms by default, and older SQLite builds
    # limit a statement to 999 bound parameters (one index plus one per key
    # column for each term).
    _UPSERT_MAX_KEY_TERMS = 400
    _UPSERT_MAX_KEY_PARAMS = 900

    def _upsert_key_batch_size(self, key_count: int) -> int:
        # Each term binds the row index plus one value per key column.
        per_term = key_count + 2
        return max(
            1, min(self._UPSERT_MAX_KEY_TERMS, self._UPSERT_MAX_KEY_PARAMS // per_term)
        )

    def _upsert_many_chunk(
        self,
        rows: Sequence[WriteRow],
        keys: Sequence[str],
        types: dict[str, ColumnType] | None,
    ) -> None:
        """Process one chunk of :py:meth:`upsert_many` as a group."""
        key_columns = [self._get_column_name(k) for k in keys]
        if (
            not key_columns
            or len(set(key_columns)) != len(key_columns)
            or not all(self.has_column(column) for column in key_columns)
        ):
            # Without key columns every row would fall through to an insert,
            # exactly as in the per-row implementation.
            for row in rows:
                self.upsert(row, keys, ensure=False)
            return

        batch: list[MutableRow] = []
        for row in rows:
            synced = self._sync_columns(row, ensure=False, types=types)
            if self._is_upsert_batchable(synced, key_columns):
                batch.append(synced)
            else:
                # NULL keys, in-row operator values and rows missing a key
                # must keep the exact per-row semantics, including whether an
                # insert generates a non-NULL key. Flush the surrounding batch
                # and process this row on its own, preserving input order.
                self._upsert_many_batch(batch, key_columns)
                batch = []
                self.upsert(synced, keys, ensure=False)
        self._upsert_many_batch(batch, key_columns)

    @staticmethod
    def _is_upsert_batchable(row: MutableRow, key_columns: Sequence[str]) -> bool:
        for column in key_columns:
            if column not in row:
                return False
            value = row[column]
            if value is None or isinstance(value, (list, tuple, set, dict)):
                return False
        return True

    def _upsert_many_batch(
        self, rows: Sequence[MutableRow], key_columns: Sequence[str]
    ) -> None:
        """Coalesce and write a set of rows whose keys are all non-NULL."""
        if not rows:
            return

        order: list[tuple[SQLWriteValue, ...]] = []
        payloads: dict[tuple[SQLWriteValue, ...], MutableRow] = {}
        for row in rows:
            key = tuple(row[column] for column in key_columns)
            payload = payloads.get(key)
            if payload is None:
                payload = {}
                payloads[key] = payload
                order.append(key)
            for name, value in row.items():
                if name not in key_columns:
                    payload[name] = value

        existing = self._existing_upsert_keys(key_columns, order)

        updates: dict[
            tuple[str, ...], list[tuple[tuple[SQLWriteValue, ...], MutableRow]]
        ] = {}
        insert_order: list[tuple[tuple[str, ...], MutableRow]] = []
        for index, key in enumerate(order):
            payload = payloads[key]
            if index in existing:
                # An update with no non-key fields only checks for a match.
                if payload:
                    fields = tuple(payload.keys())
                    updates.setdefault(fields, []).append((key, payload))
            else:
                full = dict(zip(key_columns, key, strict=True))
                full.update(payload)
                insert_order.append((tuple(full.keys()), full))

        self._execute_upsert_updates(key_columns, updates)
        self._execute_upsert_inserts(insert_order)

    def _existing_upsert_keys(
        self,
        key_columns: Sequence[str],
        keys: Sequence[tuple[SQLWriteValue, ...]],
    ) -> set[int]:
        """Return indices of key tuples that already have a matching row.

        Only the keys supplied by the caller are queried; the table is not
        loaded into memory.
        """
        existing: set[int] = set()
        batch_size = self._upsert_key_batch_size(len(key_columns))
        for start in range(0, len(keys), batch_size):
            batch = keys[start : start + batch_size]
            parts: list[Any] = []
            for offset, key in enumerate(batch):
                columns: list[Any] = [
                    expression.literal(start + offset).label("__dataset_upsert_idx")
                ]
                for position, (column, value) in enumerate(
                    zip(key_columns, key, strict=True)
                ):
                    columns.append(
                        bindparam(
                            f"__dataset_upsert_key_{offset}_{position}",
                            value=value,
                            type_=self.table.c[column].type,
                        ).label(f"__dataset_upsert_key_{position}")
                    )
                parts.append(select(*columns))

            derived = union_all(*parts).subquery()
            condition = and_(
                True,
                *[
                    self.table.c[column].is_not_distinct_from(
                        derived.c[f"__dataset_upsert_key_{position}"]
                    )
                    for position, column in enumerate(key_columns)
                ],
            )
            stmt = select(derived.c["__dataset_upsert_idx"]).select_from(
                derived.join(self.table, condition)
            )
            for result in self.db.executable.execute(stmt):
                existing.add(int(result[0]))
        return existing

    def _execute_upsert_updates(
        self,
        key_columns: Sequence[str],
        updates: dict[
            tuple[str, ...], list[tuple[tuple[SQLWriteValue, ...], MutableRow]]
        ],
    ) -> None:
        key_binds = [f"__dataset_upsert_key_{i}" for i in range(len(key_columns))]
        for fields, entries in updates.items():
            clause = and_(
                True,
                *[
                    self.table.c[column] == bindparam(bind)
                    for column, bind in zip(key_columns, key_binds, strict=True)
                ],
            )
            value_binds = [f"__dataset_upsert_value_{i}" for i in range(len(fields))]
            values: dict[str, Any] = {
                column: bindparam(bind)
                for column, bind in zip(fields, value_binds, strict=True)
            }
            stmt = self.table.update().where(clause).values(values)
            params: list[MutableRow] = []
            for key, payload in entries:
                row: MutableRow = dict(zip(key_binds, key, strict=True))
                row.update(
                    {
                        bind: payload[column]
                        for column, bind in zip(fields, value_binds, strict=True)
                    }
                )
                params.append(row)
            self.db.executable.execute(stmt, params)
            self.db._auto_commit()

    def _execute_upsert_inserts(
        self,
        inserts: Sequence[tuple[tuple[str, ...], MutableRow]],
    ) -> None:
        # Group consecutive rows with the same field set so one statement can
        # carry an executemany batch, without reordering inserts relative to
        # each other (this keeps auto-increment key assignment in input order).
        pending: list[tuple[tuple[str, ...], MutableRow]] = []
        pending_fields: tuple[str, ...] | None = None
        for fields, row in inserts:
            if pending and fields != pending_fields:
                assert pending_fields is not None
                self._execute_upsert_insert_group(pending, pending_fields)
                pending = []
            if not pending:
                pending_fields = fields
            pending.append((fields, row))
        if pending and pending_fields is not None:
            self._execute_upsert_insert_group(pending, pending_fields)

    def _execute_upsert_insert_group(
        self,
        inserts: Sequence[tuple[tuple[str, ...], MutableRow]],
        fields: Sequence[str],
    ) -> None:
        binds = [f"__dataset_upsert_insert_{i}" for i in range(len(fields))]
        stmt = self.table.insert().values(
            {
                column: bindparam(bind)
                for column, bind in zip(fields, binds, strict=True)
            }
        )
        params: list[MutableRow] = [
            {bind: row[column] for column, bind in zip(fields, binds, strict=True)}
            for _unused, row in inserts
        ]
        self.db.executable.execute(stmt, params)
        self.db._auto_commit()

    def delete(self, *clauses: ColumnElement[bool], **filters: SQLWriteValue) -> bool:
        """Delete rows from the table.

        Keyword arguments can be used to add column-based filters. The filter
        criterion will always be equality:
        ::

            table.delete(place='Berlin')

        If no arguments are given, all records are deleted.
        """
        if not self.exists:
            return False
        clause = self._args_to_clause(filters, clauses=clauses)
        stmt = self.table.delete().where(clause)
        rp = self.db.executable.execute(stmt)
        self.db._auto_commit()
        return rp.rowcount > 0

    def _reflect_table(self) -> None:
        """Load the tables definition from the database."""
        with self.db.lock:
            self._columns = None
            try:
                self._table = SQLATable(
                    self.name,
                    self.db.metadata,
                    schema=self.db.schema,
                    autoload_with=self.db.executable,
                )
            except NoSuchTableError:
                self._table = None

    def _threading_warn(self) -> None:
        if self.db.in_transaction and threading.active_count() > 1:
            warnings.warn(
                "Changing the database schema inside a transaction "
                "in a multi-threaded environment is likely to lead "
                "to race conditions and synchronization issues.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _sync_table(self, columns: Sequence[Column[Any]]) -> None:
        """Lazy load, create or adapt the table structure in the database.

        This method guarantees that self._table will be set to a non-None value
        after successful execution. If the table cannot be created or loaded,
        it raises DatasetError.
        """
        if self._table is None:
            # Load an existing table from the database.
            self._reflect_table()
        if self._table is None:
            # Create the table with an initial set of columns.
            if not self._auto_create:
                raise DatasetError(f"Table does not exist: {self.name}")
            # Keep the lock scope small because this is run very often.
            with self.db.lock:
                self._threading_warn()
                self._table = SQLATable(
                    self.name, self.db.metadata, schema=self.db.schema
                )
                if self._primary_id is not False:
                    # This can go wrong on DBMS like MySQL and SQLite where
                    # tables cannot have no columns.
                    column = Column(
                        self._primary_id,
                        self._primary_type,
                        primary_key=True,
                        autoincrement=self._primary_increment,
                    )
                    self._table.append_column(column)
                for column in columns:
                    if column.name != self._primary_id:
                        self._table.append_column(column)
                self._table.create(self.db.executable, checkfirst=True)
                self._columns = None
                self.db._auto_commit()
        elif len(columns):
            with self.db.lock:
                self._reflect_table()
                self._threading_warn()
                for column in columns:
                    if not self.has_column(column.name):
                        self.db.op.add_column(self.name, column, schema=self.db.schema)
                self._reflect_table()
                self.db._auto_commit()

    def _sync_columns(
        self,
        row: WriteRow,
        ensure: bool | None,
        types: dict[str, ColumnType] | None = None,
    ) -> MutableRow:
        """Create missing columns (or the table) prior to writes.

        If automatic schema generation is disabled (``ensure`` is ``False``),
        this will remove any keys from the ``row`` for which there is no
        matching column.
        """
        ensure = self._check_ensure(ensure)
        types = types or {}
        types = {self._get_column_name(k): v for (k, v) in types.items()}
        out = {}
        sync_columns = {}
        for name, value in row.items():
            name = self._get_column_name(name)
            if self.has_column(name):
                out[name] = value
            elif ensure:
                _type = types.get(name)
                if _type is None:
                    _type = self.db.types.guess(value)
                sync_columns[name] = Column(name, _type)
                out[name] = value
        self._sync_table(list(sync_columns.values()))
        return out

    def _check_ensure(self, ensure: bool | None) -> bool:
        if ensure is None:
            return self.db.ensure_schema
        return ensure

    def _generate_clause(
        self, column: str, op: str, value: SQLWriteValue
    ) -> ColumnElement[bool]:
        if op in ("like",):
            return self.table.c[column].like(value)
        if op in ("ilike",):
            return self.table.c[column].ilike(value)
        if op in ("notlike",):
            return self.table.c[column].notlike(value)
        if op in ("notilike",):
            return self.table.c[column].notilike(value)
        if op in (">", "gt"):
            return self.table.c[column] > value
        if op in ("<", "lt"):
            return self.table.c[column] < value
        if op in (">=", "gte"):
            return self.table.c[column] >= value
        if op in ("<=", "lte"):
            return self.table.c[column] <= value
        if op in ("=", "==", "is"):
            return self.table.c[column] == value
        if op in ("!=", "<>", "not"):
            return self.table.c[column] != value
        if op in ("in",):
            if not isinstance(value, (list, tuple, set)):
                raise QueryError(f"'in' filter requires a list, got {type(value)}")
            return self.table.c[column].in_(value)
        if op in ("notin",):
            if not isinstance(value, (list, tuple, set)):
                raise QueryError(f"'notin' filter requires a list, got {type(value)}")
            return self.table.c[column].notin_(value)
        if op in ("between", ".."):
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise QueryError("'between' filter requires a list of two values")
            start, end = value
            return self.table.c[column].between(start, end)
        if op in ("startswith",):
            if not isinstance(value, str):
                raise QueryError("'startswith' filter requires a string")
            return self.table.c[column].startswith(value, autoescape=True)
        if op in ("endswith",):
            if not isinstance(value, str):
                raise QueryError("'endswith' filter requires a string")
            return self.table.c[column].endswith(value, autoescape=True)
        return false()

    def _args_to_clause(
        self,
        args: MutableRow,
        clauses: Iterable[ColumnElement[bool]] = (),
    ) -> ColumnElement[bool]:
        clauses = list(clauses)
        for column, value in args.items():
            column = self._get_column_name(column)
            if not self.has_column(column):
                clauses.append(false())
            elif isinstance(value, (list, tuple, set)):
                clauses.append(self._generate_clause(column, "in", value))
            elif isinstance(value, dict):
                for op, op_value in value.items():
                    clauses.append(self._generate_clause(column, op, op_value))
            else:
                clauses.append(self._generate_clause(column, "=", value))
        return and_(True, *clauses)

    def _args_to_order_by(
        self, order_by: str | Sequence[str] | None
    ) -> list[UnaryExpression[Any]]:
        orderings: list[UnaryExpression[Any]] = []
        for ordering in ensure_strings(order_by):
            if ordering is None:
                continue
            column = ordering.lstrip("-")
            column = self._get_column_name(column)
            if not self.has_column(column):
                continue
            if ordering.startswith("-"):
                orderings.append(self.table.c[column].desc())
            else:
                orderings.append(self.table.c[column].asc())
        return orderings

    def _keys_to_args(
        self, row: WriteRow, keys: Sequence[str]
    ) -> tuple[MutableRow, MutableRow]:
        keys = [self._get_column_name(k) for k in ensure_strings(keys)]
        row_ = dict(row)
        args = {k: row_.pop(k, None) for k in keys}
        return args, row_

    def create_column(
        self, name: str, type: ColumnType, **kwargs: object  # noqa: A002
    ) -> None:
        """Create a new column ``name`` of a specified type.
        ::

            table.create_column('created_at', db.types.datetime)

        `type` corresponds to an SQLAlchemy type as described by
        `dataset.db.Types`. Additional keyword arguments are passed
        to the constructor of `Column`, so that default values, and
        options like `nullable` and `unique` can be set.
        ::

            table.create_column('key', unique=True, nullable=False)
            table.create_column('food', default='banana')
        """
        name = self._get_column_name(name)
        if self.has_column(name):
            log.debug(f"Column exists: {name}")
            return
        self._sync_table((Column(name, type, **kwargs),))  # type: ignore[arg-type]
        self.db._auto_commit()

    def create_column_by_example(self, name: str, value: SQLWriteValue) -> None:
        """
        Explicitly create a new column ``name`` with a type that is appropriate
        to store the given example ``value``.  The type is guessed in the same
        way as for the insert method with ``ensure=True``.
        ::

            table.create_column_by_example('length', 4.2)

        If a column of the same name already exists, no action is taken, even
        if it is not of the type we would have created.
        """
        type_ = self.db.types.guess(value)
        self.create_column(name, type_)

    def drop_column(self, name: str) -> None:
        """
        Drop the column ``name``.
        ::

            table.drop_column('created_at')

        """
        if self.db.engine is None:
            raise RuntimeError("Cannot drop columns when no engine is available.")
        if self.db.engine.dialect.name == "sqlite":
            raise RuntimeError("SQLite does not support dropping columns.")
        name = self._get_column_name(name)
        with self.db.lock:
            if not self.exists or not self.has_column(name):
                log.debug("Column does not exist: %s", name)
                return

            self._threading_warn()
            self.db.op.drop_column(self.table.name, name, schema=self.table.schema)
            self._reflect_table()
            self.db._auto_commit()

    def drop(self) -> None:
        """Drop the table from the database.

        Deletes both the schema and all the contents within it.
        """
        with self.db.lock:
            if self.exists:
                self._threading_warn()
                self.table.drop(self.db.executable, checkfirst=True)
                self._table = None
                self._columns = None
                self.db._tables.pop(self.name, None)
                self.db._auto_commit()

    def has_index(self, columns: Iterable[str]) -> bool:
        """Check if an index exists to cover the given ``columns``."""
        if not self.exists:
            return False
        columns_ = {self._get_column_name(c) for c in ensure_strings(columns)}
        if columns_ in self._indexes:
            return True
        for column in columns_:
            if not self.has_column(column):
                return False
        indexes = self.db.inspect.get_indexes(self.name, schema=self.db.schema)
        for index in indexes:
            idx_columns = index.get("column_names", [])
            if len(columns_.intersection(idx_columns)) == len(columns_):
                self._indexes.append(columns_)
                return True
        if self.table.primary_key is not None:
            pk_columns = [c.name for c in self.table.primary_key.columns]
            if len(columns_.intersection(pk_columns)) == len(columns_):
                self._indexes.append(columns_)
                return True
        return False

    def create_index(
        self, columns: Sequence[str], name: str | None = None, **kw: object
    ) -> None:
        """Create an index to speed up queries on a table.

        If no ``name`` is given a random name is created.
        ::

            table.create_index(['name', 'country'])
        """
        columns = [self._get_column_name(c) for c in ensure_strings(columns)]
        with self.db.lock:
            if not self.exists:
                raise DatasetError("Table has not been created yet.")

            for column in columns:
                if not self.has_column(column):
                    return

            if not self.has_index(columns):
                self._threading_warn()
                name = name or index_name(self.name, columns)
                columns_ = [self.table.c[c] for c in columns]

                # MySQL crashes out if you try to index very long text fields,
                # apparently. This defines (a somewhat random) prefix that
                # will be captured by the index, after which I assume the engine
                # conducts a more linear scan:
                mysql_length = {}
                for col in columns_:
                    if isinstance(col.type, MYSQL_LENGTH_TYPES):
                        mysql_length[col.name] = 10
                kw["mysql_length"] = mysql_length

                idx = Index(name, *columns_, **kw)  # type: ignore[arg-type]
                idx.create(self.db.executable)
                self.db._auto_commit()

    def find(
        self,
        *_clauses: ColumnElement[bool],
        _limit: int | None = None,
        _offset: int = 0,
        order_by: str | Sequence[str] | None = None,
        _streamed: bool = False,
        _step: int | None = QUERY_STEP,
        **kwargs: SQLWriteValue,
    ) -> ResultIter:
        """Perform a simple search on the table.

        Simply pass keyword arguments as ``filter``.
        ::

            results = table.find(country='France')
            results = table.find(country='France', year=1980)

        Using ``_limit``::

            # just return the first 10 rows
            results = table.find(country='France', _limit=10)

        You can sort the results by single or multiple columns. Append a minus
        sign to the column name for descending order::

            # sort results by a column 'year'
            results = table.find(country='France', order_by='year')
            # return all rows sorted by multiple columns (descending by year)
            results = table.find(order_by=['country', '-year'])

        You can also submit filters based on criteria other than equality,
        see :ref:`advanced_filters` for details.

        To run more complex queries with JOINs, or to perform GROUP BY-style
        aggregation, you can also use :py:meth:`db.query() <dataset.Database.query>`
        to run raw SQL queries instead.
        """
        if not self.exists:
            return ResultIter(None, row_type=self.db.row_type)
        if self.db.engine is None:
            raise RuntimeError("Cannot run queries when no engine is available.")

        if _step is False or _step == 0:
            _step = None

        orderings = self._args_to_order_by(order_by)
        args = self._args_to_clause(kwargs, clauses=_clauses)
        query = self.table.select().where(args).limit(_limit).offset(_offset)
        if len(orderings):
            query = query.order_by(*orderings)

        stream_conn = None
        conn = self.db.executable
        if _streamed:
            stream_conn = self.db.engine.connect()
            conn = stream_conn.execution_options(stream_results=True)

        return ResultIter(
            conn.execute(query),
            row_type=self.db.row_type,
            step=_step,
            connection=stream_conn,
        )

    def find_one(
        self, *args: ColumnElement[bool], **kwargs: SQLWriteValue
    ) -> OutRow | None:
        """Get a single result from the table.

        Works just like :py:meth:`find() <dataset.Table.find>` but returns one
        result, or ``None``.
        ::

            row = table.find_one(country='United States')
        """
        if not self.exists:
            return None

        resiter = self.find(*args, _limit=1, _step=None, **kwargs)  # type: ignore[arg-type]
        try:
            for row in resiter:
                return row
        finally:
            resiter.close()
        return None

    def count(self, *_clauses: ColumnElement[bool], **kwargs: SQLWriteValue) -> int:
        """Return the count of results for the given filter set."""
        # NOTE: this does not have support for limit and offset since I can't
        # see how this is useful. Still, there might be compatibility issues
        # with people using these flags. Let's see how it goes.
        if not self.exists:
            return 0

        args = self._args_to_clause(kwargs, clauses=_clauses)
        query = select(func.count()).where(args)
        query = query.select_from(self.table)
        rp = self.db.executable.execute(query)
        res = rp.fetchone()
        if res is not None:
            return int(res[0])
        return 0

    def __len__(self) -> int:
        """Return the number of rows in the table."""
        return self.count()

    def distinct(
        self,
        *args: str | ColumnElement[bool],
        _limit: int | None = None,
        _offset: int | None = 0,
        **kwargs: SQLWriteValue,
    ) -> ResultIter:
        """Return all the unique (distinct) values for the given ``columns``.
        ::

            # returns only one row per year, ignoring the rest
            table.distinct('year')
            # works with multiple columns, too
            table.distinct('year', 'country')
            # you can also combine this with a filter
            table.distinct('year', country='China')
        """
        if not self.exists:
            return ResultIter(None, row_type=self.db.row_type)

        columns = []
        clauses = []
        for column in args:
            if isinstance(column, ClauseElement):
                clauses.append(column)
            else:
                if not self.has_column(column):
                    raise DatasetError(f"No such column: {column}")
                columns.append(self.table.c[column])

        clause = self._args_to_clause(kwargs, clauses=clauses)
        if not len(columns):
            return ResultIter(None, row_type=self.db.row_type)

        q = (
            expression.select(*columns)
            .distinct()
            .where(clause)
            .limit(_limit)
            .offset(_offset)
            .order_by(*[c.asc() for c in columns])
        )
        return self.db.query(q)

    # Legacy methods for running find queries.
    all = find

    def __iter__(self) -> ResultIter:
        """Return all rows of the table as simple dictionaries.

        Allows for iterating over all rows in the table without explicitly
        calling :py:meth:`find() <dataset.Table.find>`.
        ::

            for row in table:
                print(row)
        """
        return self.find()

    def __repr__(self) -> str:
        """Get table representation."""
        return f"<Table({self.table.name})>"
