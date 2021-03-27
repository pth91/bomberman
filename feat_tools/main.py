import featuretools as ft
from featuretools import EntitySet as es
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


def prepare_EntitySet():
    # we start w/ only one round of logged games
    # read the pickle that was saved after one round of playing the game
    game_state = pd.read_pickle("./game_states_logging/game_state_round-1")

    # to build entities and their realtions we need to extract information
    # from game_state

    common_columns = ["round", "step"]

    # we need a data frame for the field in each iteration
    field_df = game_state[common_columns + ["field"]]

    # we need a data frame for the bombs in each iteration
    bombs_df = game_state[common_columns + ["bombs"]]

    # we need a data frame for the agents own information
    self_df = game_state[common_columns + ["self"]]

    # we need a (temporary) data frame for the opposing agents
    opposing_df = game_state[common_columns + ["others"]]

    # we need a data frame for the coins and explosion maps
    coins_df = game_state[common_columns + ["coins"]]
    explosions_df = game_state[common_columns + ["explosion_map"]]

    # we have to drop duplicates in each of the data frames
    # NOTE: pandas is not able to find duplicates if columns consist of list
    # types, this includes numpy arrays. hence we have to drop duplicates by
    # subsetting on the round and step columns.
    field_df.drop_duplicates(subset=common_columns, inplace=True)
    bombs_df.drop_duplicates(subset=common_columns, inplace=True)
    self_df.drop_duplicates(subset=common_columns, inplace=True)
    opposing_df.drop_duplicates(subset=common_columns, inplace=True)
    coins_df.drop_duplicates(subset=common_columns, inplace=True)
    explosions_df.drop_duplicates(subset=common_columns, inplace=True)

    # to spare ram and for convenience we will define entity sets at this
    # point and will drop the dataframes after importing the data to the
    # EntitySet object of feature tools
    bomberman_entity = es(id="bomberman")

    # at this point these entity objects are still empty and we have to
    # supply data
    bomberman_entity = bomberman_entity.entity_from_dataframe(
        entity_id="field", dataframe=field_df, index="step"
    )
    bomberman_entity = bomberman_entity.entity_from_dataframe(
        entity_id="bombs", dataframe=field_df, index="step"
    )
    bomberman_entity = bomberman_entity.entity_from_dataframe(
        entity_id="self", dataframe=self_df, index="step"
    )
    bomberman_entity = bomberman_entity.entity_from_dataframe(
        entity_id="opposing", dataframe=opposing_df, index="step"
    )
    bomberman_entity = bomberman_entity.entity_from_dataframe(
        entity_id="coins", dataframe=coins_df, index="step"
    )
    bomberman_entity = bomberman_entity.entity_from_dataframe(
        entity_id="explosions", dataframe=explosions_df, index="step"
    )
    return bomberman_entity


def run_tools():
    bomberman_entity = prepare_EntitySet()
    print("ha")


if __name__ == "__main__":
    run_tools()
