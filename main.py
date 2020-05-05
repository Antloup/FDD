# import pandas as pd
import gzip
import statistics
from collections import defaultdict
import random

import unicodecsv as csv
from sklearn.metrics import f1_score, precision_recall_fscore_support
import warnings
from sklearn.model_selection import cross_val_score
import itertools
from typing import List
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

directory = "./input"
delimitor: str = ','
train_file = '/TRAIN.CSV.gz'
test_file = '/TEST.CSV.gz'

input_list: List[str] = ['Base', 's', 'SingleMineral', 'hotkey00', 'hotkey01', 'hotkey02', 'hotkey10', 'hotkey11',
                         'hotkey12', 'hotkey20', 'hotkey21', 'hotkey22', 'hotkey30', 'hotkey31', 'hotkey32', 'hotkey40',
                         'hotkey41', 'hotkey42', 'hotkey50', 'hotkey51', 'hotkey52', 'hotkey60', 'hotkey61', 'hotkey62',
                         'hotkey70', 'hotkey71', 'hotkey72', 'hotkey80', 'hotkey81', 'hotkey82', 'hotkey90', 'hotkey91',
                         'hotkey92']

flatten = lambda x: [item for sublist in x for item in sublist]

targeted_player_rate: float = .01
minimum_accepted_proba: float = .3
relevant_percentage: float = .0
minimum_diff_proba = .1
late_players: List[str] = []
late_player_number: int= 1500
split: float = 0.8
targeted_player: List[str] = []
targeted_player_confusion = defaultdict(list)  # Key : target, Value : predicted
becareful_player_confusion = defaultdict(list)  # Key : predicted, Value : target
player_game_dict = defaultdict(list)  # Key : Player, Value : List of his games
confusion_decision_tree = defaultdict()  # Key : predicted, Value : Decision tree
confusion_features = defaultdict()  # Key : predicted, Value : features


class Action():
    BASE = 0
    SELECT = 1
    MINERAL = 2
    HOTKEY = 3

    def __init__(self, input: str):
        self.value: int = 0
        try:
            self.value: int = input_list.index(input)
        except ValueError:
            print('Unknow Key')
        self.type: int = self.input_of(input)

    def input_of(self, input: str) -> int:
        if input[0] == 'h':
            return Action.HOTKEY
        elif input[0] == 's':
            return Action.SELECT
        elif input[0] == 'S':
            return Action.MINERAL
        else:
            return Action.BASE


def action_to_key(inputs: List[Action]) -> str:
    key: str = ''
    for i in inputs:
        key += chr(ord('0') + i.value)
    return key


class Race():
    UNK = 0
    PROTOSS = 1
    TERRAN = 2
    ZERG = 3

    def __init__(self, race: str):
        self.value = self.race_of(race)

    def race_of(self, race: str) -> int:
        if race == "Protoss":
            return Race.PROTOSS
        elif race == "Terran":
            return Race.TERRAN
        elif race == "Zerg":
            return Race.ZERG
        else:
            return Race.UNK


class Game:
    def __init__(self, player: str, race: str, inputs: str):
        self.player: str = player
        self.race: Race = Race(race)
        self.inputs: List[List[Action]] = []
        input_sequence: List[Action] = []
        while len(inputs) != 0:
            action = inputs[:inputs.find(delimitor)]
            if action[0] == 't':
                self.inputs.append(input_sequence.copy())
                input_sequence = []
            else:
                input_sequence.append(Action(action))
            inputs = inputs[len(action) + 1:]
        self.inputs.append(input_sequence)

    def __str__(self):
        return "Player:" + self.player + "\t Race:" + str(self.race.value) + " \t Nb of input set:" + str(
            len(self.inputs))


class Features:

    def __init__(self, games: List[Game] = None,
                 hp_observed_time: int = 15,
                 hp_first_move: int = 1,
                 hp_action_order: int = 5,
                 key_comb_2_time: int = 25,
                 hp_window=None,
                 hp_key_comb_time: int = 25,
                 hp_key_comb_mean: int = 10,
                 hp_key_comb_std: int = 2):
        if hp_window is None:
            hp_window = [2, 3, 4]
        self.hp_observed_time: int = hp_observed_time
        self.hp_first_move: int = hp_first_move
        self.hp_action_order: int = hp_action_order
        self.hp_window: List[int] = hp_window
        self.hp_key_comb_time: int = hp_key_comb_time
        self.hp_key_comb_mean: int = hp_key_comb_mean
        self.hp_key_comb_std: int = hp_key_comb_std
        self.hp_key_comb_2_time: int = key_comb_2_time
        self.key_comb_list: List[str] = []
        self.time_range: List[List[int]] = [
            [0, self.hp_key_comb_time],
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10],
            [0, 2], [2, 4], [4, 6], [6, 8], [8, 10], [10, 12], [12, 14], [14, 16], [16, 18], [18, 20],
            [0, 5], [5, 10], [10, 15], [15, 20], [20, 25], [25, 30], [30, 35],
            [0, 10], [10, 20], [20, 30], [30, 40],
            [0, 20], [20, 40]
        ]

    def get_features(self, games: List[Game]) -> List[List[int]]:
        features: List[List[int]] = []
        for g in games:
            features.append(self.extract_features(g))
        return features

    def get_features2(self, games: List[Game]) -> List[List[int]]:
        features: List[List[int]] = []
        for g in games:
            features.append(self.extract_features2(g))
        return features

    def extract_features(self, game: Game) -> List[int]:
        features_list: List[List[int]] = [
            # [game.race.value], # Not used anymore
            self.first_move(game),
            self.num_first_move(game),
            self.num_key_type(game),
            self.key_combinaison(game),
            # self.analyse_key_comb(game),
            # self.late_move(game), # Not revelant
            # self.spam(game) # Not (very) revelant (doesn't affect kaggle score)
        ]
        return [item for sublist in features_list for item in sublist]

    def extract_features2(self, game: Game) -> List[int]:
        features_list: List[List[int]] = [
            self.first_move(game),
            self.num_first_move(game),
            self.num_key_type(game),
            self.key_combinaison(game),
            self.analyse_key_comb(game),
        ]
        return [item for sublist in features_list for item in sublist]

    def num_first_move(self, game: Game, param_observed_time=None) -> List[int]:
        score: int = 0
        observed_time: int = self.hp_observed_time
        if param_observed_time is not None:
            observed_time: int = observed_time
        for i in range(min(observed_time, len(game.inputs))):
            score += len(game.inputs[i])
        return [score]

    def late_move(self, game: Game) -> List[int]:
        if len(game.inputs[-1]) == 0:
            return [-1]
        return [game.inputs[-1][-1].value]

    def spam(self, game: Game) -> List[int]:
        max_spam: List[int] = [0] * len(input_list)
        last_key: int = -1
        current_spam: int = 1
        for i in range(min(self.hp_observed_time, len(game.inputs))):
            for j in game.inputs[i]:
                if j.value == last_key:
                    current_spam += 1
                else:
                    max_spam[last_key] = max(current_spam, max_spam[last_key])
                    current_spam = 1
                    last_key = j.value
        return max_spam

    def game_length(self, game: Game) -> List[int]:
        return [len(game.inputs)]

    def game_race(self, game: Game) -> List[int]:
        return [game.race.value]

    def num_key_type(self, game: Game) -> List[int]:
        action_num: List[int] = [0] * len(input_list)
        action_order: List[int] = [0] * self.hp_action_order
        action_order_list: List[int] = [0] * len(input_list)
        k: int = 0
        # action_type: List[int] = [0, 0, 0, 0]
        # action_used: List[int] = [0] * len(input_list)
        for i in range(min(self.hp_observed_time, len(game.inputs))):
            for j in game.inputs[i]:
                action_num[j.value] += 1
                # action_type[j.type] += 1
                # action_used[j.value] = 1
                if action_order_list[j.value] == 0 and k < self.hp_action_order:
                    action_order[k] = j.value
                    k += 1
                    action_order_list[j.value] = 1

        return action_num + action_order

    def first_move(self, game: Game) -> List[int]:
        if len(game.inputs[0]) == 0:
            return [-1] * self.hp_first_move
        else:
            score: List[int] = [0] * self.hp_first_move
            for i in range(min(self.hp_first_move - 1, len(game.inputs[0]) - 1)):
                score[i] = game.inputs[0][i].value
            return score

    def key_combinaison(self, game: Game) -> List[int]:
        combinaison: List[List[int]] = [[0] * len(input_list) for i in input_list]
        game_input = flatten(game.inputs[:min(self.hp_key_comb_2_time, len(game.inputs))])
        for v, w in zip(game_input[:-1], game_input[1:]):
            combinaison[int(v.value)][int(w.value)] += 1
        return flatten(combinaison)

    def analyse_key_comb(self, game: Game, time_range=None) -> List[int]:
        if time_range is None:
            time_range = [[0, self.hp_key_comb_time]]
        key_score: List[int] = [0] * len(self.key_comb_list)
        j: int = -1
        for tr in time_range:
            j += 1
            if len(game.inputs) <= tr[0]:
                break
            inputs = flatten(game.inputs[tr[0]:min(tr[1], len(game.inputs))])
            for key_comb in self.key_comb_list:
                window = len(key_comb)
                for i in range(min(self.hp_key_comb_time - window, len(game.inputs) - window)):
                    key = action_to_key(inputs[i:i + window])
                    if key == key_comb:
                        key_score[j] += 1

        return key_score

    def most_relevant_key_combinaison(self, games: List[Game]) -> List[str]:
        dictionnary: dict = {}
        games_len: int = len(games)
        j: int = -1
        for game in games:
            j += 1
            inputs = flatten(game.inputs[0:min(self.hp_key_comb_time, len(game.inputs))])
            for window in self.hp_window:
                for i in range(len(inputs) - window):
                    key = action_to_key(inputs[i:i + window])
                    if dictionnary.get(key) is None:
                        dictionnary[key] = [0] * games_len
                        dictionnary[key][j] = 1
                    else:
                        dictionnary[key][j] += 1
        return self.relevant_function(dictionnary)

    def relevant_function(self, dictionnary: dict) -> List[str]:
        key_comb: List[str] = []
        for key, value in dictionnary.items():
            if sum(x > 0 for x in value) >= relevant_percentage * len(value):
                key_comb.append(key)
        return key_comb

    def get_relevant_key_combinaison(self, games: List[Game]) -> List[str]:
        key_comb: List[List[str]] = []  # TODO:Put mean as reference ?
        player_game_dict = defaultdict(list)
        for game in games:
            player_game_dict[game.player].append(game)

        for player in player_game_dict:
            if player in targeted_player:
                key_comb.append(self.most_relevant_key_combinaison(player_game_dict[player]))

        flatten_key_comb = list(set(flatten(key_comb)))  # Flatten / remove duplicate
        return flatten_key_comb


def get_games(data, label: bool = True) -> List[Game]:
    games: List[Game] = []
    i :int = 0
    for row in data:
        i += 1
        data_decode = row.decode("utf-8")
        if i > late_player_number:
            late_players.append(data_decode[:data_decode.find(delimitor)])
        player = ''
        if label:
            player = data_decode[:data_decode.find(delimitor)]
        data_decode = data_decode[len(player) + 1:]
        race = data_decode[:data_decode.find(delimitor)]
        inputs = data_decode[len(race) + 1:]
        games.append(Game(player, race, inputs))
    return games


def init_targeted_players(y_test, y_pred, params):
    class_report = classification_report(y_test, y_pred, output_dict=True)
    player_score = [0, 0, 0]
    for player, score in class_report.items():
        if score['f1-score'] == 1:
            player_score[0] += 1
        elif score['f1-score'] > targeted_player_rate:
            player_score[1] += 1
        else:
            player_score[2] += 1
            targeted_player.append(player)
    print(player_score)
    for i in range(len(y_test)):
        if y_test[i] in targeted_player and y_test[i] != y_pred[i]:
            targeted_player_confusion[y_test[i]].append(y_pred[i])
            becareful_player_confusion[y_pred[i]].append(y_test[i])

    for key, value in targeted_player_confusion.items():
        targeted_player_confusion[key] = list(set(value))
    for key, value in becareful_player_confusion.items():
        becareful_player_confusion[key] = list(set(value))

    for prediction in becareful_player_confusion.keys():
        players_poll_games: List[Game] = []
        players_poll_games += player_game_dict[prediction]
        for player in becareful_player_confusion[prediction]:
            players_poll_games += player_game_dict[player]
        confusion_features[prediction] = Features(games, params[1], params[2], params[3],
                                                  params[4], params[5], params[6], params[7])
        confusion_features[prediction].key_comb_list = confusion_features[prediction].get_relevant_key_combinaison(
            players_poll_games)
        confusion_decision_tree[prediction] = ensemble.RandomForestClassifier(n_estimators=100, random_state=0,
                                                                              class_weight='balanced')
        confusion_decision_tree[prediction] = confusion_decision_tree[prediction].fit(
            confusion_features[prediction].get_features2(players_poll_games),
            [g.player for g in players_poll_games])


if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    data_train = gzip.open(directory + train_file, 'rb')
    games: List[Game] = get_games(data_train, label=True)
    late_players = list(set(late_players))
    late_players_game: List[Game]= []

    for game in games:
        player_game_dict[game.player].append(game)
        if game.player in late_players:
            late_players_game.append(game)
    games = late_players_game

    params_benchmark: bool = False
    testing: bool = False # Fine tuning params
    result_params = [90, 25, 5, 12, 25, [2, 3, 4, 5, 6, 7, 8, 9], 25, 0, 0]

    # params = [number of estimators,
    #  observed time for key sum, number of first move to observe, number of action to take in order,
    #  key combinaison 2 time, key combinaison window (list), key combinaison time, key comb mean, key comb std]
    params = [[90], [25], [5], [12], [25], [[2, 3, 4, 5, 6]], [25], [5], [3]]

    if params_benchmark:
        params = [[90], [25], [5], [12], [25], [[0]], [25], [1], [3]]

    best_conf: List[int] = []
    best_score: float = .0

    if testing:
        for hyper_params in itertools.product(*params):
            clf = ensemble.RandomForestClassifier(n_estimators=hyper_params[0], random_state=0, class_weight='balanced')
            features: Features = Features(games, hyper_params[1], hyper_params[2], hyper_params[3], hyper_params[4],
                                          hyper_params[5], hyper_params[6], hyper_params[7])
            random.seed(42)
            random.shuffle(games)
            first_forest_features = features.get_features(games)
            X_train = first_forest_features[0: int(len(games) * split)]
            X_test = first_forest_features[int(len(games) * split) + 1: len(games)]
            y_train = [g.player for g in games[0: int(len(games) * split)]]
            y_test = [g.player for g in games[int(len(games) * split) + 1: len(games)]]
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            if not params_benchmark:
                print(classification_report(y_test, y_pred))
                print(clf.feature_importances_)

                # print(targeted_player_confusion)
                # print(becareful_player_confusion)

                ########## Start : Trying subtree ##########
                # becareful_players: List[str] = []
                # for target in targeted_player:
                #     print('The target: ' + target + '\n')
                #     for g in player_game_dict[target]:
                #         game_feature = np.asarray(features.extract_features(g)).reshape(1, -1)
                #         if clf.predict(game_feature) != target:
                #             print('Prediction failed : \n')
                #             for player_choice in np.where(clf.predict_proba(game_feature) > 0.1)[1]:
                #                 becareful_players.append(clf.classes_[player_choice])
                #                 print(clf.classes_[player_choice])
                #             print(clf.predict(game_feature))
                # becareful_players = list(set(becareful_players))

                #####
                # mrkc_list: List[str] = []
                # for key, value in targeted_player_confusion.items():
                #     targeted_player_mrkc = features.most_relevant_key_combinaison(player_game_dict[key])
                #     mix_player_mrkc = []
                #     for key2 in value:
                #         mix_player_mrkc.append(features.most_relevant_key_combinaison(player_game_dict[key2]))
                #     mix_player_mrkc = flatten(mix_player_mrkc)
                #     final_mrkc = []
                #     for combi in targeted_player_mrkc:
                #         if combi not in mix_player_mrkc:
                #             final_mrkc.append(combi)
                #     mrkc_list += final_mrkc
                # features.key_comb_list = mrkc_list
                #####

                nb_notbc_ok = 0
                nb_notbc_ko = 0
                nb_bc_ok = 0
                nb_bc_ok_same = 0
                nb_bc_ok_diff = 0
                nb_bc_ko = 0
                nb_bc_ko_in_list = 0
                nb_bc_ko_wtf = 0
                current = -1
                for x in range(len(X_test)):
                    current += 1
                    prediction = clf.predict(np.asarray(X_test[x]).reshape(1, -1))[0]
                    prediction_proba = clf.predict_proba(np.asarray(X_test[x]).reshape(1, -1))
                    if prediction not in list(becareful_player_confusion.keys()):
                    # if prediction not in list(becareful_player_confusion.keys()) or \
                    #         len(np.where(prediction_proba > minimum_accepted_proba)[1]) == 1:
                        if prediction == y_test[x]:
                            nb_notbc_ok += 1
                        else:
                            nb_notbc_ko += 1
                    else:
                        print(current / len(X_test))
                        # players_poll_games: List[Game] = []
                        # players_poll_games += player_game_dict[prediction]
                        # for player in becareful_player_confusion[prediction]:
                        #     players_poll_games += player_game_dict[player]
                        # if games[current] in players_poll_games:
                        #     players_poll_games.remove(games[current])
                        # features.key_comb_list = features.get_relevant_key_combinaison(players_poll_games)
                        # clf2 = ensemble.RandomForestClassifier(n_estimators=100, random_state=0,
                        #                                        class_weight='balanced')
                        # clf2 = clf2.fit(features.get_features2(players_poll_games),
                        #                 [g.player for g in players_poll_games])
                        # new_prediction = clf2.predict(np.asarray(features.extract_features2(games[current])).reshape(1, -1))[0]
                        new_prediction = confusion_decision_tree[prediction].predict(
                            np.asarray(confusion_features[prediction].extract_features2(games[current])).reshape(1,
                                                                                                                 -1))[0]

                        new_prediction_proba = confusion_decision_tree[prediction].predict_proba(
                            np.asarray(confusion_features[prediction].extract_features2(games[current])).reshape(1,
                                                                                                                 -1))[0]
                        # new_prediction_proba.sort(reverse=True)
                        new_prediction_proba = -np.sort(-new_prediction_proba)
                        if True:
                        # if(new_prediction_proba[0] - new_prediction_proba[1] < minimum_diff_proba):
                            new_prediction = prediction

                        # print(classification_report(y_test2, y_pred2))
                        if new_prediction == y_test[x]:
                            nb_bc_ok += 1
                            if prediction == new_prediction:
                                nb_bc_ok_same += 1
                                # print('nb_bc_ok_same')
                            else:
                                nb_bc_ok_diff += 1
                                # print('nb_bc_ok_diff')
                            # print('nb_bc_ok')
                        elif prediction == y_test[x]:
                            nb_bc_ko_wtf += 1
                            # print('nb_bc_wtf')
                        else:
                            nb_bc_ko += 1
                            # print('nb_bc_ko')
                            if prediction in targeted_player_confusion[y_test[x]]:
                                nb_bc_ko_in_list += 1
                                # print('was in confusion list')
                print('nb_notbc_ok \t' + str(nb_notbc_ok))
                print('nb_notbc_ko \t' + str(nb_notbc_ko))
                print('nb_bc_ok \t' + str(nb_bc_ok))
                print('nb_bc_ok_same \t' + str(nb_bc_ok_same))
                print('nb_bc_ok_diff \t' + str(nb_bc_ok_diff))
                print('nb_bc_ko \t' + str(nb_bc_ko))
                print('nb_bc_ko_in_list \t' + str(nb_bc_ko_in_list))
                print('nb_bc_ko_wtf \t' + str(nb_bc_ko_wtf))
                print('score \t' + str(nb_bc_ok_diff - nb_bc_ko_wtf))
                ########## START : Trying mrkc splitting ##########
                # mrkc_list: List[str] = []
                # for key, value in targeted_player_confusion.items():
                #     targeted_player_mrkc = features.most_relevant_key_combinaison(player_game_dict[key])
                #     mix_player_mrkc = []
                #     for key2 in value:
                #         mix_player_mrkc.append(features.most_relevant_key_combinaison(player_game_dict[key2]))
                #     mix_player_mrkc = flatten(mix_player_mrkc)
                #     final_mrkc = []
                #     for combi in targeted_player_mrkc:
                #         if combi not in mix_player_mrkc:
                #             final_mrkc.append(combi)
                #     mrkc_list += final_mrkc
                # features.key_comb_list = mrkc_list
                #
                # clf2 = ensemble.RandomForestClassifier(n_estimators=150, random_state=0,
                #                                        class_weight='balanced')
                # X_train, X_test, y_train, y_test = train_test_split(features.get_features2(games),
                #                                                     [g.player for g in games],
                #                                                     test_size=0.4, shuffle=True, random_state=0)
                # clf2 = clf2.fit(X_train, y_train)
                # y_pred = clf2.predict(X_test)
                # print(classification_report(y_test, y_pred))
                # class_report = classification_report(y_test, y_pred, output_dict=True)
                # player_score = [0, 0, 0]
                # for player, score in class_report.items():
                #     if score['f1-score'] == 1:
                #         player_score[0] += 1
                #     elif score['f1-score'] > targeted_player_rate:
                #         player_score[1] += 1
                #     else:
                #         player_score[2] += 1
                #         targeted_player.append(player)
                # print(player_score)
                ########## END : Trying mrkc splitting ##########

            if params_benchmark:
                print(hyper_params)
                print(classification_report(y_test, y_pred))
                print(clf.feature_importances_)
                print(f1_score(y_test, y_pred, average='micro'))
                if f1_score(y_test, y_pred, average='micro') > best_score:
                    best_score = f1_score(y_test, y_pred, average='micro')
                    best_conf = hyper_params

        if params_benchmark:
            print('Best :')
            print(best_conf)
            print(best_score)

    else:
        clf = ensemble.RandomForestClassifier(n_estimators=result_params[0], class_weight='balanced')
        features: Features = Features(games, result_params[1], result_params[2], result_params[3])
        clf = clf.fit(features.get_features(games), [g.player for g in games])
        data_train = gzip.open(directory + test_file, 'rb')
        games: List[Game] = get_games(data_train, label=False)
        y_pred = clf.predict(features.get_features(games))

        with open('result_f1.csv', 'wb') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',', encoding='utf-8')
            i: int = 1
            csv_writer.writerow(['RowId'] + ['prediction'])
            for prediction in y_pred:
                csv_writer.writerow([str(i)] + [prediction])
                i += 1
