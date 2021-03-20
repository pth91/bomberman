import featuretools as ft
import pandas as pd
import numpy as np


def log_gamestates(round, agents_df):
    """
    function is used to log the game state and is called from the bomberman
    main.py after each round is finished
    """
    for it in range(0, len(agents_df)):
        agents_df[it].to_pickle(
            f"./feat_tools/game_states_logging/game_state_round-{round}",
        )
        print(type(agents_df[it].iloc[0, 3]))


def run_tools():
    # we start w/ only one round of logged games
    game_state = pd.read_csv("./game_states_logging/game_state_round-1")
    # game state has duplicates which we have to drop
    game_state.drop_duplicates(inplace=True)
    print("ha")


if __name__ == "__main__":
    run_tools()
