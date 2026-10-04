"""
api/accounts/db.py  -  SQLAlchemy 引擎/会话/Base，本仓库第一个真正的数据库层。

只服务账号/鉴权/配额相关的表（api/accounts/models_db.py）——现有的
api/store.py / api/conversation_store.py / api/dataset_store.py 仍然是
进程内存态，不迁移到这里，两者故意分开。
"""

from __future__ import annotations

from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from api.settings import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Iterator[Session]:
    """FastAPI Depends() 用：每个请求一个会话，用完关闭。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
