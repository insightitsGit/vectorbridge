from typing import Iterator
import numpy as np

from .base import VectorConnector, VectorRecord, ConnectorConfig


class PgvectorConnector(VectorConnector):
    """Read/write vectors from a PostgreSQL + pgvector table."""

    def connect(self) -> None:
        import psycopg2
        import psycopg2.extras
        self._conn = psycopg2.connect(
            host=self.config.host,
            port=self.config.port,
            dbname=self.config.database,
            user=self.config.user,
            password=self.config.password,
        )
        self._conn.autocommit = False

    def disconnect(self) -> None:
        if self._conn:
            self._conn.close()

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.config.table}")
            return cur.fetchone()[0]

    def read_batches(self, batch_size: int = 256, offset: int = 0) -> Iterator[list[VectorRecord]]:
        meta_cols = self.config.metadata_columns
        meta_select = (", " + ", ".join(meta_cols)) if meta_cols else ""
        sql = (
            f"SELECT {self.config.id_column}, {self.config.vector_column}{meta_select} "
            f"FROM {self.config.table} "
            f"ORDER BY {self.config.id_column} "
            f"LIMIT %s OFFSET %s"
        )
        cursor_offset = offset
        while True:
            with self._conn.cursor() as cur:
                cur.execute(sql, (batch_size, cursor_offset))
                rows = cur.fetchall()
            if not rows:
                break
            batch = []
            for row in rows:
                vid = str(row[0])
                # pgvector returns a list or string depending on driver
                raw = row[1]
                if isinstance(raw, str):
                    raw = [float(x) for x in raw.strip("[]").split(",")]
                vec = np.array(raw, dtype=np.float32)
                meta = {col: row[2 + i] for i, col in enumerate(meta_cols)}
                batch.append(VectorRecord(id=vid, vector=vec, metadata=meta))
            yield batch
            cursor_offset += batch_size

    def write_batch(self, records: list[VectorRecord]) -> int:
        import psycopg2.extras
        meta_cols = self.config.metadata_columns
        if meta_cols:
            cols = f"{self.config.id_column}, {self.config.vector_column}, {', '.join(meta_cols)}"
            placeholders = "%s, %s::vector" + ", %s" * len(meta_cols)
        else:
            cols = f"{self.config.id_column}, {self.config.vector_column}"
            placeholders = "%s, %s::vector"

        sql = (
            f"INSERT INTO {self.config.table} ({cols}) VALUES ({placeholders}) "
            f"ON CONFLICT ({self.config.id_column}) DO UPDATE "
            f"SET {self.config.vector_column} = EXCLUDED.{self.config.vector_column}"
        )
        rows = []
        for r in records:
            vec_str = "[" + ",".join(str(x) for x in r.vector.tolist()) + "]"
            row = [r.id, vec_str] + [r.metadata.get(c) for c in meta_cols]
            rows.append(row)
        with self._conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, rows)
        self._conn.commit()
        return len(records)

    def create_index(self, dimension: int, **kwargs) -> None:
        with self._conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.table} (
                    {self.config.id_column} TEXT PRIMARY KEY,
                    {self.config.vector_column} vector({dimension})
                )
            """)
        self._conn.commit()

    def supports_incremental(self) -> bool:
        return True

    def read_changes(self, since_checkpoint: str, batch_size: int = 256) -> Iterator[list[VectorRecord]]:
        """Requires a created_at or updated_at column named in metadata_columns."""
        sql = (
            f"SELECT {self.config.id_column}, {self.config.vector_column} "
            f"FROM {self.config.table} "
            f"WHERE updated_at > %s ORDER BY updated_at LIMIT %s"
        )
        offset = 0
        while True:
            with self._conn.cursor() as cur:
                cur.execute(sql, (since_checkpoint, batch_size))
                rows = cur.fetchall()
            if not rows:
                break
            batch = []
            for row in rows:
                raw = row[1]
                if isinstance(raw, str):
                    raw = [float(x) for x in raw.strip("[]").split(",")]
                vec = np.array(raw, dtype=np.float32)
                batch.append(VectorRecord(id=str(row[0]), vector=vec))
            yield batch
            offset += batch_size
