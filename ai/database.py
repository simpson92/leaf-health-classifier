from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import DATABASE_URL, DB_POOL_SIZE

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=DB_POOL_SIZE, max_overflow=10)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def init_db():
    from . import models  # noqa: F401  -- 테이블 등록
    Base.metadata.create_all(bind=engine)