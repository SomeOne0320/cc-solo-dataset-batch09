# Local SQLite evaluation environment

Use the standard Linux ARM64/Python3.11 model image. No external database service, cloud credential or remote destination is needed. Runtime and test tools are locked in requirements-cc-solo.txt. Tests default to local in-memory SQLite; lifecycle tests may use temporary SQLite files.

```bash
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/dataset-get-pip.py
python3 /tmp/dataset-get-pip.py --user --break-system-packages pip==24.3.1
python3 -m pip install --user --break-system-packages -r requirements-cc-solo.txt
python3 -m pip install --user --break-system-packages --no-build-isolation --no-deps -e .
python3 -m pytest -q test/test_database.py test/test_table.py
python3 -m mypy --strict dataset
```

The initial version passes the selected SQLite tests. Under these pinned tools, the initial strict Mypy check reports one existing no-untyped-call diagnostic for SQLAlchemy supports_sane_rowcount in dataset/table.py. Keep this baseline issue separate from any newly introduced type errors. The repository's strict typing and compatibility rules still apply to changed code. Do not report PostgreSQL or MySQL integration as verified without running those services; this task needs only local SQLite. Do not change the package version or publish a package release.
