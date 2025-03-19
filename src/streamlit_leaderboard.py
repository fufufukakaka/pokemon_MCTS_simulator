import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
import pytz

# 今回は簡略化のために database_handler をベタ書きする
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


# Set page configuration
st.set_page_config(
    page_title="Pokémon Trainer Leaderboard", page_icon="🏆", layout="wide"
)

# Title and description
st.title("🏆 Pokémon Trainer Leaderboard")
st.markdown("### Ranking of trainers based on simulation battles")

# Initialize database connection
db_handler = DatabaseHandler()

# Get leaderboard data
leaderboard_data = db_handler.get_leaderboard_data()
df_leaderboard = pd.DataFrame(leaderboard_data)

# Display leaderboard with styling
if not df_leaderboard.empty:
    # Add medal emojis for top 3
    def get_medal(position):
        if position == 1:
            return "🥇"
        elif position == 2:
            return "🥈"
        elif position == 3:
            return "🥉"
        else:
            return ""

    df_leaderboard["Medal"] = df_leaderboard["position"].apply(get_medal)

    # Reorder columns for display
    df_display = df_leaderboard[["position", "Medal", "name", "rating", "rank"]]
    df_display = df_display.rename(
        columns={
            "position": "Position",
            "name": "Trainer Name",
            "rating": "Rating",
            "rank": "Original Rank",
        }
    )

    # Show the leaderboard table
    st.dataframe(
        df_display.style.background_gradient(subset=["Rating"], cmap="viridis"),
        use_container_width=True,
        hide_index=True,
    )

    # Create columns for visualization and stats
    col1, col2 = st.columns([3, 1])

    with col1:
        # Bar chart of top trainers
        top_trainers = df_leaderboard.head(10).copy()
        chart = (
            alt.Chart(top_trainers)
            .mark_bar()
            .encode(
                x=alt.X("rating:Q", title="Rating"),
                y=alt.Y("name:N", sort="-x", title="Trainer"),
                color=alt.Color("rating:Q", scale=alt.Scale(scheme="viridis")),
                tooltip=["name", "rating"],
            )
            .properties(title="Top 10 Trainers by Rating", height=400)
        )
        st.altair_chart(chart, use_container_width=True)

    with col2:
        # Show stats
        st.subheader("Leaderboard Stats")
        st.metric("Number of Trainers", len(df_leaderboard))
        st.metric("Highest Rating", df_leaderboard["rating"].max())
        st.metric("Average Rating", round(df_leaderboard["rating"].mean(), 1))
        st.metric(
            "Rating Range",
            f"{df_leaderboard['rating'].max() - df_leaderboard['rating'].min()}",
        )

    # Show recent battles
    st.subheader("Recent Battles")
    battle_history = db_handler.load_battle_history()

    if battle_history:
        battle_data = []
        for battle in battle_history[-10:]:  # Get last 10 battles
            battle_data.append(
                {
                    "Trainer A": battle.trainer_a_name,
                    "Rating A": battle.trainer_a_rating,
                    "Trainer B": battle.trainer_b_name,
                    "Rating B": battle.trainer_b_rating,
                    "Time": battle.log_saved_time,
                }
            )

        battle_df = pd.DataFrame(battle_data)
        st.dataframe(battle_df, use_container_width=True, hide_index=True)
    else:
        st.info("No battle history available yet.")
else:
    st.error(
        "No leaderboard data available. Please make sure the database is properly initialized."
    )

# Footer
st.markdown("---")
st.markdown("*Last updated: {}*".format(datetime.now().astimezone(pytz.timezone('Asia/Tokyo')).strftime("%Y-%m-%d %H:%M:%S")))
st.markdown("*Last updated: {}*".format(datetime.now().astimezone(timezone('Asia/Tokyo')).strftime("%Y-%m-%d %H:%M:%S")))
