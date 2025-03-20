import os

from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.models import Trainer

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


class TrainerRating(Base):
    __tablename__ = "trainer_rating"
    id = Column(Integer, primary_key=True, index=True)
    rank = Column(Integer)
    name = Column(String)
    sim_rating = Column(Integer)


class DatabaseHandler:
    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def get_session(self):
        return self.SessionLocal()

    def initialize_battle_history(self):
        """
        対戦履歴テーブルを truncate する
        """
        session = self.get_session()
        session.query(BattleHistory).delete()
        session.commit()
        session.close()

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

    def create_rating_table(
        self,
        trainers: list[Trainer],
        default_rating: int = 1500,
    ):
        """
        truncate してから rank, name, rating(初期値 1500) を登録する
        """
        session = self.get_session()
        session.query(TrainerRating).delete()
        session.commit()

        for trainer in trainers:
            trainer_rating = TrainerRating(
                rank=trainer.rank,
                name=trainer.name,
                sim_rating=default_rating,
            )
            session.add(trainer_rating)
        session.commit()
        session.close()

    def update_trainer_rating(self, rank: int, rating: int):
        """
        rank に対応するトレーナーの rating を更新する
        """
        session = self.get_session()
        trainer = session.query(TrainerRating).filter_by(rank=rank).first()
        if trainer:
            trainer.sim_rating = rating
            session.commit()
        session.close()

    def load_trainer_ratings(self) -> list[Trainer]:
        """
        トレーナーのレーティングをデータベースから取得する
        """
        session = self.get_session()
        trainer_ratings = (
            session.query(TrainerRating).order_by(TrainerRating.rank).all()
        )
        trainers = [
            Trainer(
                name=trainer.name,
                rank=trainer.rank,
                rating=trainer.sim_rating,
                pokemons=[],
                raw_pokemons=[],
            )
            for trainer in trainer_ratings
        ]
        session.close()
        return trainers

    def load_battle_history(self) -> list[BattleHistory]:
        """
        対戦履歴をデータベースから取得する
        """
        session = self.get_session()
        battle_history = session.query(BattleHistory).all()
        session.close()
        return battle_history

    def get_leaderboard_data(self) -> list[dict]:
        """
        トレーナーのレーティングをレーティングの高い順にソートして返す
        レーダーボード表示に適したデータ構造で返す
        """
        session = self.get_session()
        trainer_ratings = (
            session.query(TrainerRating).order_by(TrainerRating.sim_rating.desc()).all()
        )
        leaderboard = []
        position = 1

        for trainer in trainer_ratings:
            leaderboard.append(
                {
                    "position": position,
                    "rank": trainer.rank,
                    "name": trainer.name,
                    "rating": trainer.sim_rating,
                }
            )
            position += 1

        session.close()
        return leaderboard
