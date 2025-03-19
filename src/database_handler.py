import os

from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
DATABASE_URL = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:"
    f"{POSTGRES_PORT}/{POSTGRES_DB}"
)


class BattleHistory(Base):
    __tablename__ = "battle_history"
    id = Column(Integer, primary_key=True, index=True)
    trainer_a_name = Column(String)
    trainer_b_name = Column(String)
    trainer_a_rating = Column(Integer)
    trainer_b_rating = Column(Integer)
    log_saved_time = Column(String)


class DatabaseHandler:
    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def get_session(self):
        return self.SessionLocal()

    def insert_battle_history(
        self,
        trainer_a_name: str,
        trainer_b_name: str,
        trainer_a_rating: int,
        trainer_b_rating: int,
        log_saved_time: str,
    ):
        session = self.get_session()
        battle_history = BattleHistory(
            trainer_a_name=trainer_a_name,
            trainer_b_name=trainer_b_name,
            trainer_a_rating=trainer_a_rating,
            trainer_b_rating=trainer_b_rating,
            log_saved_time=log_saved_time,
        )
        session.add(battle_history)
        session.commit()
        session.close()
