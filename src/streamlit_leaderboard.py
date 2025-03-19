import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timezone

from src.database_handler import DatabaseHandler

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
# datetime is JST format
st.markdown("*Last updated: {}*".format(datetime.now().astimezone(timezone('Asia/Tokyo')).strftime("%Y-%m-%d %H:%M:%S")))
