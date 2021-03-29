import featuretools as ft
from featuretools import EntitySet as es
from featuretools import Relationship as rs
from featuretools.primitives import AggregationPrimitive, TransformPrimitive
from featuretools.variable_types import Numeric
import pandas as pd
import numpy as np


# class ManhattenDistance(TransformPrimitive):
#    name = "ManhattanDistance"
#    input_types = [Numeric, Numeric]
#    return_type = [Numeric, Numeric]
#
#    def get_function(self):
#        def ManhattanDistance(numeric, numeric):
#            return


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


def opposing_agents_generate_features(
    self_df, opposing_0, opposing_1, opposing_2
):
    # create entity set
    opposing_agents_EntitySet = es(id="bomberman")
    # at this point these entity objects are still empty and we have to
    # supply data
    opposing_agents_EntitySet = (
        opposing_agents_EntitySet.entity_from_dataframe(
            entity_id="self", dataframe=self_df, index="step_self"
        )
    )
    opposing_agents_EntitySet = (
        opposing_agents_EntitySet.entity_from_dataframe(
            entity_id="opposing_0",
            dataframe=opposing_0,
            index="step_opposing_0",
        )
    )
    opposing_agents_EntitySet = (
        opposing_agents_EntitySet.entity_from_dataframe(
            entity_id="opposing_1",
            dataframe=opposing_1,
            index="step_opposing_1",
        )
    )
    opposing_agents_EntitySet = (
        opposing_agents_EntitySet.entity_from_dataframe(
            entity_id="opposing_2",
            dataframe=opposing_2,
            index="step_opposing_2",
        )
    )

    # add relations

    opposing_relationship_0 = rs(
        opposing_agents_EntitySet["self"]["step_self"],
        opposing_agents_EntitySet["opposing_0"]["step_self"],
    )
    opposing_relationship_1 = rs(
        opposing_agents_EntitySet["self"]["step_self"],
        opposing_agents_EntitySet["opposing_1"]["step_self"],
    )
    opposing_relationship_2 = rs(
        opposing_agents_EntitySet["self"]["step_self"],
        opposing_agents_EntitySet["opposing_2"]["step_self"],
    )
    opposing_agents_EntitySet = opposing_agents_EntitySet.add_relationship(
        opposing_relationship_0
    )
    # opposing_agents_EntitySet = opposing_agents_EntitySet.add_relationship(
    #    opposing_relationship_1
    # )
    # opposing_agents_EntitySet = opposing_agents_EntitySet.add_relationship(
    #    opposing_relationship_2
    # )

    feature_matrix, feature_defs = ft.dfs(
        entityset=opposing_agents_EntitySet,
        target_entity="opposing_0",
        agg_primitives=["count", "sum"],
        max_depth=2,
    )

    return feature_matrix, feature_defs


def prepare_EntitySet():
    """
    function reads the pickle file from a run of one round of the
    bombermangame and initializes the data in the form of an
    featuretools.EntitySet
    """
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
    # renaming has to be done for functionality in feature set, furthermore
    # we have to define one (self) to many relations (the rest of the game),
    # hence we add the self_df['step_self'] column st. realtions can be
    # constructed between tables
    self_df.drop_duplicates(subset=common_columns, inplace=True)
    self_df.rename(columns={"step": "step_self"}, inplace=True)
    self_df["name"] = self_df.loc[:, "self"].str[0]
    self_df["score"] = self_df.loc[:, "self"].str[1]
    self_df["bomb_available"] = self_df.loc[:, "self"].str[2]
    self_df["location"] = self_df.loc[:, "self"].str[3]
    self_df["location"] = self_df["location"].apply(lambda x: np.array(x))
    self_df.drop("self", inplace=True, axis=1)
    self_df.reset_index(inplace=True, drop=True)

    field_df.drop_duplicates(subset=common_columns, inplace=True)
    field_df.rename(columns={"step": "step_field"}, inplace=True)
    field_df["step_self"] = self_df["step_self"]

    bombs_df.drop_duplicates(subset=common_columns, inplace=True)
    bombs_df.rename(columns={"step": "step_bombs"}, inplace=True)
    bombs_df["step_self"] = self_df["step_self"]

    opposing_df.drop_duplicates(subset=common_columns, inplace=True)
    opposing_df.rename(columns={"step": "step_opposing"}, inplace=True)
    opposing_df["step_self"] = self_df["step_self"]
    opposing_df = opposing_df.explode("others", ignore_index=True)
    opposing_agents = opposing_df.loc[:, "others"].str[0].unique()
    opposing_0 = opposing_df[
        opposing_df.loc[:, "others"].str[0] == opposing_agents[0]
    ]
    opposing_0["name"] = opposing_0.loc[:, "others"].str[0]
    opposing_0["score"] = opposing_0.loc[:, "others"].str[1]
    opposing_0["bomb_available"] = opposing_0.loc[:, "others"].str[2]
    opposing_0["location"] = opposing_0.loc[:, "others"].str[3]
    opposing_0["location"] = opposing_0["location"].apply(
        lambda x: np.array(x)
    )
    opposing_0.drop("others", inplace=True, axis=1)
    opposing_0.rename(
        columns={"step_opposing": "step_opposing_0"}, inplace=True
    )
    opposing_0["step_self"].astype(int)
    # opposing_0["step_opposing_0"].astype(int)
    opposing_0.reset_index(inplace=True, drop=True)
    opposing_1 = opposing_df[
        opposing_df.loc[:, "others"].str[0] == opposing_agents[1]
    ]
    opposing_1["name"] = opposing_1.loc[:, "others"].str[0]
    opposing_1["score"] = opposing_1.loc[:, "others"].str[1]
    opposing_1["bomb_available"] = opposing_1.loc[:, "others"].str[2]
    opposing_1["location"] = opposing_1.loc[:, "others"].str[3]
    opposing_1["location"] = opposing_1["location"].apply(
        lambda x: np.array(x)
    )
    opposing_1.drop("others", inplace=True, axis=1)
    opposing_1.rename(
        columns={"step_opposing": "step_opposing_1"}, inplace=True
    )
    opposing_2 = opposing_df[
        opposing_df.loc[:, "others"].str[0] == opposing_agents[2]
    ]
    opposing_2["name"] = opposing_2.loc[:, "others"].str[0]
    opposing_2["score"] = opposing_2.loc[:, "others"].str[1]
    opposing_2["bomb_available"] = opposing_2.loc[:, "others"].str[2]
    opposing_2["location"] = opposing_2.loc[:, "others"].str[3]
    opposing_2["location"] = opposing_2["location"].apply(
        lambda x: np.array(x)
    )
    opposing_2.drop("others", inplace=True, axis=1)
    opposing_2.rename(
        columns={"step_opposing": "step_opposing_2"}, inplace=True
    )

    coins_df.drop_duplicates(subset=common_columns, inplace=True)
    coins_df.rename(columns={"step": "step_coins"}, inplace=True)
    coins_df["step_self"] = self_df["step_self"]

    explosions_df.drop_duplicates(subset=common_columns, inplace=True)
    explosions_df.rename(columns={"step": "step_explosions"}, inplace=True)
    explosions_df["step_self"] = self_df["step_self"]

    # to spare ram and for convenience we will define entity sets at this
    # point and will drop the dataframes after importing the data to the
    # EntitySet object of feature tools
    bomberman_entity = es(id="bomberman")

    opposing_agents_features = opposing_agents_generate_features(
        self_df, opposing_0, opposing_1, opposing_2
    )
    # at this point these entity objects are still empty and we have to
    # supply data
    bomberman_entity = bomberman_entity.entity_from_dataframe(
        entity_id="field", dataframe=field_df, index="step_field"
    )
    # bomberman_entity = bomberman_entity.entity_from_dataframe(
    #    entity_id="bombs", dataframe=bombs_df, index="step_bombs"
    # )
    bomberman_entity = bomberman_entity.entity_from_dataframe(
        entity_id="self", dataframe=self_df, index="step_self"
    )
    bomberman_entity = bomberman_entity.entity_from_dataframe(
        entity_id="opposing_0", dataframe=opposing_0, index="step_opposing_0"
    )
    bomberman_entity = bomberman_entity.entity_from_dataframe(
        entity_id="opposing_1", dataframe=opposing_1, index="step_opposing_1"
    )
    bomberman_entity = bomberman_entity.entity_from_dataframe(
        entity_id="opposing_2", dataframe=opposing_2, index="step_opposing_2"
    )
    # bomberman_entity = bomberman_entity.entity_from_dataframe(
    #    entity_id="coins", dataframe=coins_df, index="step_coins"
    # )
    # bomberman_entity = bomberman_entity.entity_from_dataframe(
    #    entity_id="explosions",
    #    dataframe=explosions_df,
    #    index="step_explosions",
    # )
    return opposing_agents_features


def run_tools():
    # we want to start by getting an entity set from one run of the
    # bomberman game
    opposing_agents_features = prepare_EntitySet()

    ## next, we want to add realtionships between the columns of the entities
    # field_relationship = rs(
    #    bomberman_entity["self"]["step_self"],
    #    bomberman_entity["field"]["step_self"],
    # )
    ## bombs_relationship = rs(
    ##    bomberman_entity["self"]["step_self"],
    ##    bomberman_entity["bombs"]["step_self"],
    ## )
    # opposing_relationship_0 = rs(
    #    bomberman_entity["self"]["step_self"],
    #    bomberman_entity["opposing_0"]["step_self"],
    # )
    # opposing_relationship_1 = rs(
    #    bomberman_entity["self"]["step_self"],
    #    bomberman_entity["opposing_1"]["step_self"],
    # )
    # opposing_relationship_2 = rs(
    #    bomberman_entity["self"]["step_self"],
    #    bomberman_entity["opposing_2"]["step_self"],
    # )
    ## then we have to add above relationship to the entity set
    ## bomberman_entity = bomberman_entity.add_relationship(field_relationship)
    ## bomberman_entity = bomberman_entity.add_relationship(bombs_relationship)
    # bomberman_entity = bomberman_entity.add_relationship(
    #    opposing_relationship_0
    # )
    # bomberman_entity = bomberman_entity.add_relationship(
    #    opposing_relationship_1
    # )
    # bomberman_entity = bomberman_entity.add_relationship(
    #    opposing_relationship_2
    # )
    print("ha")


if __name__ == "__main__":
    run_tools()
